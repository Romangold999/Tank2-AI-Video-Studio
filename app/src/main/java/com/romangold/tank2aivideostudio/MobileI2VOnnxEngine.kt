package com.romangold.tank2aivideostudio

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.net.Uri
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.security.SecureRandom
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Local MobileI2V pipeline: LTX image encoder -> MobileDiT Flow Euler -> Turbo-VAED decoder.
 * Runs on CPU through ONNX Runtime. This is intentionally conservative about session lifetime:
 * encoder is closed before the denoiser is created, and denoiser before decoder.
 */
class MobileI2VOnnxEngine(
    private val modelFiles: MobileI2VModelManager.ModelFiles,
    private val onProgress: (String, Int) -> Unit = { _, _ -> }
) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val random = SecureRandom()
    private var activeSession: OrtSession? = null
    private var activeOptions: OrtSession.SessionOptions? = null

    fun generate(contentResolver: ContentResolver, imageUri: Uri): List<Bitmap> {
        val source = decodeSourceImage(contentResolver, imageUri)
        val inputImage = try { fitImage(source, WIDTH, HEIGHT) } finally { source.recycle() }
        try {
            onProgress("Кодирование изображения", 3)
            val guide = encodeImage(inputImage)
            onProgress("Подготовка MobileI2V", 8)
            val latents = denoise(guide)
            onProgress("Декодирование видео", 82)
            return decodeVideo(latents)
        } finally {
            inputImage.recycle()
        }
    }

    private fun encodeImage(bitmap: Bitmap): FloatArray {
        val session = openSession(modelFiles.encoder)
        try {
            check(session.inputNames == setOf("image")) { "Неожиданные входы VAE encoder: ${session.inputNames}" }
            val imageData = bitmapToNcthw(bitmap)
            val tensor = tensor(imageData, longArrayOf(1, 3, 1, HEIGHT.toLong(), WIDTH.toLong()))
            tensor.use {
                session.run(mapOf("image" to it)).use { result ->
                    val moments = (result[0] as OnnxTensor).floatBuffer
                    val mean = FloatArray(LATENT_CHANNELS * LATENT_H * LATENT_W)
                    val logvar = FloatArray(mean.size)
                    for (i in mean.indices) {
                        mean[i] = moments.get(i)
                        logvar[i] = moments.get(mean.size + i).coerceIn(-30f, 20f)
                    }
                    val guide = FloatArray(mean.size)
                    for (i in guide.indices) {
                        val noise = gaussian()
                        guide[i] = (mean[i] + exp(0.5f * logvar[i]) * noise) * MOBILEI2V_VAE_SCALE
                    }
                    return guide
                }
            }
        } finally { closeSession() }
    }

    private fun denoise(guide: FloatArray): FloatArray {
        val session = openSession(modelFiles.denoiser)
        try {
            val expected = setOf("latent", "timestep", "cond_mask", "flow_score")
            check(session.inputNames == expected) { "Неожиданные входы MobileDiT: ${session.inputNames}" }
            val latentSize = LATENT_CHANNELS * LATENT_T * LATENT_H * LATENT_W
            var latents = FloatArray(latentSize) { gaussian() }
            clampGuide(latents, guide)
            val condMask = FloatArray(LATENT_T * LATENT_H * LATENT_W)
            for (i in 0 until LATENT_H * LATENT_W) condMask[i] = 1f
            val maskTensor = tensor(condMask, longArrayOf(1, condMask.size.toLong()))
            val scoreTensor = tensor(floatArrayOf(2f), longArrayOf(1))
            maskTensor.use { mask ->
                scoreTensor.use { score ->
                    for (step in 0 until STEPS) {
                        val sigma = shiftedSigma(1f - step.toFloat() / STEPS)
                        val sigmaNext = shiftedSigma(1f - (step + 1).toFloat() / STEPS)
                        val latentTensor = tensor(latents, longArrayOf(1, LATENT_CHANNELS.toLong(), LATENT_T.toLong(), LATENT_H.toLong(), LATENT_W.toLong()))
                        val timeTensor = tensor(floatArrayOf(sigma * 1000f), longArrayOf(1))
                        latentTensor.use { latent ->
                            timeTensor.use { time ->
                                session.run(mapOf(
                                    "latent" to latent,
                                    "timestep" to time,
                                    "cond_mask" to mask,
                                    "flow_score" to score
                                )).use { result ->
                                    val prediction = (result[0] as OnnxTensor).floatBuffer
                                    val updated = FloatArray(latentSize)
                                    val delta = sigmaNext - sigma
                                    for (i in updated.indices) updated[i] = latents[i] + prediction.get(i) * delta
                                    latents = updated
                                }
                            }
                        }
                        clampGuide(latents, guide)
                        onProgress("MobileDiT: шаг ${step + 1}/$STEPS", 8 + ((step + 1) * 68 / STEPS))
                    }
                }
            }
            return latents
        } finally { closeSession() }
    }

    private fun decodeVideo(latents: FloatArray): List<Bitmap> {
        val session = openSession(modelFiles.decoder)
        try {
            check(session.inputNames == setOf("latents")) { "Неожиданные входы Turbo-VAED: ${session.inputNames}" }
            val input = tensor(latents, longArrayOf(1, LATENT_CHANNELS.toLong(), LATENT_T.toLong(), LATENT_H.toLong(), LATENT_W.toLong()))
            input.use {
                session.run(mapOf("latents" to it)).use { result ->
                    val video = (result[0] as OnnxTensor).floatBuffer
                    val frames = ArrayList<Bitmap>(FRAME_COUNT)
                    val plane = DECODER_H * WIDTH
                            for (frame in 0 until FRAME_COUNT) {
                        val pixels = IntArray(WIDTH * HEIGHT)
                        for (y in 0 until HEIGHT) for (x in 0 until WIDTH) {
                            val p = y * WIDTH + x
                            // ONNX decoder output is N,C,T,H,W, not N,T,C,H,W.
                            // Each channel plane contains all 17 temporal frames.
                            val temporalOffset = frame * plane
                            val r = toByte(video.get(temporalOffset + p))
                            val g = toByte(video.get(FRAME_COUNT * plane + temporalOffset + p))
                            val b = toByte(video.get(2 * FRAME_COUNT * plane + temporalOffset + p))
                            pixels[p] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
                        }
                        frames.add(Bitmap.createBitmap(pixels, WIDTH, HEIGHT, Bitmap.Config.ARGB_8888))
                    }
                    return frames
                }
            }
        } finally { closeSession() }
    }

    private fun openSession(file: java.io.File): OrtSession {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        return try {
            env.createSession(file.absolutePath, options).also {
                activeSession = it
                activeOptions = options
            }
        } catch (t: Throwable) {
            options.close()
            throw t
        }
    }

    private fun closeSession() {
        try { activeSession?.close() } finally {
            activeSession = null
            activeOptions?.close()
            activeOptions = null
        }
    }

    private fun tensor(values: FloatArray, shape: LongArray): OnnxTensor {
        val buffer = ByteBuffer.allocateDirect(values.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        buffer.put(values).rewind()
        return OnnxTensor.createTensor(env, buffer, shape)
    }

    private fun decodeSourceImage(resolver: ContentResolver, uri: Uri): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        resolver.openInputStream(uri).use { stream ->
            requireNotNull(stream) { "Не удалось открыть изображение" }
            BitmapFactory.decodeStream(stream, null, bounds)
        }
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "Некорректное изображение" }
        var sample = 1
        while (bounds.outWidth / sample > 2560 || bounds.outHeight / sample > 1440) {
            sample *= 2
        }
        val options = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        return resolver.openInputStream(uri).use { stream ->
            requireNotNull(stream) { "Не удалось повторно открыть изображение" }
            BitmapFactory.decodeStream(stream, null, options)
                ?: error("Не удалось декодировать изображение")
        }
    }

    private fun bitmapToNcthw(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(WIDTH * HEIGHT)
        bitmap.getPixels(pixels, 0, WIDTH, 0, 0, WIDTH, HEIGHT)
        val plane = WIDTH * HEIGHT
        val out = FloatArray(3 * plane)
        for (i in pixels.indices) {
            val c = pixels[i]
            out[i] = (((c shr 16) and 255) / 127.5f) - 1f
            out[plane + i] = (((c shr 8) and 255) / 127.5f) - 1f
            out[2 * plane + i] = ((c and 255) / 127.5f) - 1f
        }
        return out
    }

    private fun fitImage(source: Bitmap, width: Int, height: Int): Bitmap {
        val output = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val scale = max(width.toFloat() / source.width, height.toFloat() / source.height)
        val scaledW = (source.width * scale).toInt()
        val scaledH = (source.height * scale).toInt()
        val left = (scaledW - width) / 2
        val top = (scaledH - height) / 2
        val canvas = android.graphics.Canvas(output)
        canvas.drawBitmap(source, null, Rect(-left, -top, scaledW - left, scaledH - top), null)
        return output
    }

    private fun clampGuide(latents: FloatArray, guide: FloatArray) {
        val frameSize = LATENT_CHANNELS * LATENT_H * LATENT_W
        System.arraycopy(guide, 0, latents, 0, frameSize)
    }

    private fun shiftedSigma(sigma: Float): Float = (SHIFT * sigma) / (1f + (SHIFT - 1f) * sigma)

    private fun gaussian(): Float {
        var u1 = random.nextDouble()
        val u2 = random.nextDouble()
        if (u1 < 1e-12) u1 = 1e-12
        return (kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * Math.PI * u2)).toFloat()
    }

    private fun toByte(value: Float): Int = (((value.coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()).coerceIn(0, 255)

    override fun close() = closeSession()

    companion object {
        const val WIDTH = 1280
        const val HEIGHT = 720
        private const val DECODER_H = 736
        private const val LATENT_CHANNELS = 128
        private const val LATENT_T = 3
        private const val LATENT_H = 23
        private const val LATENT_W = 40
        private const val FRAME_COUNT = 17
        private const val STEPS = 28
        private const val SHIFT = 3f
        private const val MOBILEI2V_VAE_SCALE = 0.41407f
    }
}
