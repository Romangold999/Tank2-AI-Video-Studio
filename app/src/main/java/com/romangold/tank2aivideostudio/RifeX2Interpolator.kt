package com.romangold.tank2aivideostudio

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.security.MessageDigest

/**
 * RIFE v4.9 x2 frame interpolation.
 *
 * Call from a background thread. For every adjacent pair of source frames this
 * class synthesizes the midpoint, returning [A, midpoint(A,B), B, ...].
 * It never changes the source frame rate metadata or duplicates frames.
 */
class RifeX2Interpolator(context: Context) : AutoCloseable {
    private val appContext = context.applicationContext
    private val modelFile = File(File(appContext.filesDir, "models"), MODEL_FILE)
    private val environment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var sessionOptions: OrtSession.SessionOptions? = null

    /**
     * Stream frames to a synchronous consumer instead of retaining every
     * generated midpoint in RAM. Original bitmaps remain owned by the caller.
     * Each generated midpoint is valid only during onFrame and is recycled
     * immediately after the callback returns. The consumer must finish reading
     * or encoding the bitmap before returning.
     */
    @Synchronized
    fun forEachInterpolatedFrame(
        sourceFrames: List<Bitmap>,
        onModelDownloadProgress: (downloaded: Long, total: Long) -> Unit = { _, _ -> },
        onFrameProgress: (completedPairs: Int, totalPairs: Int) -> Unit = { _, _ -> },
        onFrame: (frame: Bitmap, isGeneratedMidpoint: Boolean) -> Unit
    ) {
        require(sourceFrames.size >= 2) { "RIFE needs at least two source frames" }
        ensureModel(onModelDownloadProgress)
        val activeSession = session ?: createSession().also { session = it }
        val totalPairs = sourceFrames.size - 1

        onFrame(sourceFrames.first(), false)
        for (index in 0 until totalPairs) {
            val first = sourceFrames[index]
            val second = sourceFrames[index + 1]
            require(first.width == second.width && first.height == second.height) {
                "All source frames must have identical dimensions"
            }
            val midpoint = inferMidpoint(activeSession, first, second)
            try {
                onFrame(midpoint, true)
            } finally {
                midpoint.recycle()
            }
            onFrame(second, false)
            onFrameProgress(index + 1, totalPairs)
        }
        // Preserve the source clip's sample-count duration at exactly 2x FPS.
        onFrame(sourceFrames.last(), false)
    }

    @Synchronized
    fun interpolateSequence(
        sourceFrames: List<Bitmap>,
        onModelDownloadProgress: (downloaded: Long, total: Long) -> Unit = { _, _ -> },
        onFrameProgress: (completedPairs: Int, totalPairs: Int) -> Unit = { _, _ -> }
    ): List<Bitmap> {
        require(sourceFrames.size >= 2) { "RIFE needs at least two source frames" }
        ensureModel(onModelDownloadProgress)
        val activeSession = session ?: createSession().also { session = it }

        val totalPairs = sourceFrames.size - 1
        val result = ArrayList<Bitmap>(sourceFrames.size + totalPairs + 1)
        for (index in 0 until totalPairs) {
            val first = sourceFrames[index]
            val second = sourceFrames[index + 1]
            require(first.width == second.width && first.height == second.height) {
                "All source frames must have identical dimensions"
            }
            result.add(first)
            result.add(inferMidpoint(activeSession, first, second))
            onFrameProgress(index + 1, totalPairs)
        }
        result.add(sourceFrames.last())
        // A final hold frame makes 17 source samples at 8 FPS become 34 at 16 FPS.
        result.add(sourceFrames.last().copy(Bitmap.Config.ARGB_8888, false))
        return result
    }

    private fun createSession(): OrtSession {
        val options = OrtSession.SessionOptions()
        options.setIntraOpNumThreads(CPU_THREADS)
        options.setInterOpNumThreads(1)
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        val created = try {
            environment.createSession(modelFile.absolutePath, options)
        } catch (error: Exception) {
            options.close()
            throw error
        }
        try {
            val expectedInputs = setOf("img0", "img1", "timestep")
            check(created.inputNames == expectedInputs) {
                "Unexpected RIFE inputs: ${created.inputNames}"
            }
            check(created.outputNames.contains(OUTPUT_NAME)) {
                "RIFE output '$OUTPUT_NAME' not found: ${created.outputNames}"
            }
            sessionOptions = options
            return created
        } catch (error: Exception) {
            created.close()
            options.close()
            throw error
        }
    }

    private fun inferMidpoint(activeSession: OrtSession, first: Bitmap, second: Bitmap): Bitmap {
        val originalWidth = first.width
        val originalHeight = first.height
        val paddedWidth = ((originalWidth + 31) / 32) * 32
        val paddedHeight = ((originalHeight + 31) / 32) * 32
        val shape = longArrayOf(1, 3, paddedHeight.toLong(), paddedWidth.toLong())

        val firstInput = bitmapToTensorData(first, paddedWidth, paddedHeight)
        val secondInput = bitmapToTensorData(second, paddedWidth, paddedHeight)
        val timestepBuffer = directFloatBuffer(1).put(0.5f).apply { rewind() }

        OnnxTensor.createTensor(environment, firstInput, shape).use { img0 ->
            OnnxTensor.createTensor(environment, secondInput, shape).use { img1 ->
                OnnxTensor.createTensor(environment, timestepBuffer, longArrayOf(1)).use { timestep ->
                    val inputs = mapOf("img0" to img0, "img1" to img1, "timestep" to timestep)
                    activeSession.run(inputs).use { outputs ->
                        val output = outputs.get(OUTPUT_NAME).orElseThrow {
                            IllegalStateException("RIFE returned no '$OUTPUT_NAME' tensor")
                        } as OnnxTensor
                        val values = output.floatBuffer
                        val outputPixels = IntArray(originalWidth * originalHeight)
                        val planeSize = paddedWidth * paddedHeight
                        for (y in 0 until originalHeight) {
                            for (x in 0 until originalWidth) {
                                val sourceIndex = y * paddedWidth + x
                                val red = toByte(values.get(sourceIndex))
                                val green = toByte(values.get(planeSize + sourceIndex))
                                val blue = toByte(values.get(2 * planeSize + sourceIndex))
                                outputPixels[y * originalWidth + x] =
                                    (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
                            }
                        }
                        return Bitmap.createBitmap(
                            outputPixels, originalWidth, originalHeight, Bitmap.Config.ARGB_8888
                        )
                    }
                }
            }
        }
    }

    private fun bitmapToTensorData(bitmap: Bitmap, paddedWidth: Int, paddedHeight: Int): FloatBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val planeSize = paddedWidth * paddedHeight
        val tensor = directFloatBuffer(planeSize * 3)
        for (y in 0 until paddedHeight) {
            val sourceY = minOf(y, height - 1)
            for (x in 0 until paddedWidth) {
                val sourceX = minOf(x, width - 1)
                val color = pixels[sourceY * width + sourceX]
                val offset = y * paddedWidth + x
                tensor.put(offset, ((color shr 16) and 0xFF) / 255f)
                tensor.put(planeSize + offset, ((color shr 8) and 0xFF) / 255f)
                tensor.put(2 * planeSize + offset, (color and 0xFF) / 255f)
            }
        }
        tensor.position(0)
        tensor.limit(planeSize * 3)
        return tensor
    }

    private fun ensureModel(onProgress: (Long, Long) -> Unit) {
        if (modelFile.isFile && sha256(modelFile) == MODEL_SHA256) return
        modelFile.parentFile?.mkdirs()
        val temporary = File(modelFile.parentFile, "$MODEL_FILE.asset-copy")
        temporary.delete()
        appContext.assets.open("models/$MODEL_FILE").use { input ->
            temporary.outputStream().buffered().use { output ->
                val buffer = ByteArray(1024 * 256)
                var copied = 0L
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    output.write(buffer, 0, count)
                    copied += count
                    onProgress(copied, -1L)
                }
            }
        }
        val actualHash = sha256(temporary)
        check(temporary.length() == MODEL_SIZE_BYTES && actualHash == MODEL_SHA256) {
            val actualSize = temporary.length()
            temporary.delete()
            "Bundled RIFE model verification failed: expected ${MODEL_SIZE_BYTES} bytes / ${MODEL_SHA256}, got $actualSize bytes / $actualHash"
        }
        if (modelFile.exists()) modelFile.delete()
        check(temporary.renameTo(modelFile)) { "Could not install bundled RIFE model" }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(1024 * 128)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun directFloatBuffer(count: Int): FloatBuffer =
        ByteBuffer.allocateDirect(count * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()

    private fun toByte(value: Float): Int =
        (value.coerceIn(0f, 1f) * 255f + 0.5f).toInt()

    @Synchronized
    override fun close() {
        session?.close()
        session = null
        sessionOptions?.close()
        sessionOptions = null
    }

    companion object {
        const val OUTPUT_FPS_MULTIPLIER = 2
        private const val CPU_THREADS = 2
        private const val MODEL_FILE = "rife49.onnx"
        private const val MODEL_SIZE_BYTES = 21458882L
        private const val MODEL_URL = "https://huggingface.co/edgetools/rife/resolve/main/rife49.onnx"
        private const val MODEL_SHA256 = "76e4cef9ab42fa7dd4e8f6e4aba47462051e3faa969e4bca6479784fbab0ac6f"
        private const val OUTPUT_NAME = "output"
    }
}
