package com.romangold.tank2aivideostudio

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.File
import java.nio.ByteBuffer

/**
 * Streaming H.264 MP4 writer for generated bitmaps.
 *
 * Frames are converted into the flexible YUV420 input image exposed by
 * MediaCodec. The class never retains a clip in memory: each bitmap is
 * submitted and released before the next one is requested from the caller.
 */
class VideoMp4Encoder(
    private val output: File,
    private val width: Int = 1280,
    private val height: Int = 720,
    private val fps: Int = 16,
    private val bitrate: Int = 8_000_000
) {
    init {
        require(width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0)
        require(fps > 0)
    }

    fun encode(
        frames: Iterable<Bitmap>,
        onProgress: (encodedFrames: Int) -> Unit = {}
    ): File = encodeStream(
        produceFrames = { emit -> frames.forEach(emit) },
        onProgress = onProgress
    )

    /**
     * Encodes each frame immediately when the producer emits it. The producer
     * must not retain generated frames after emit returns.
     */
    fun encodeStream(
        produceFrames: (emit: (Bitmap) -> Unit) -> Unit,
        onProgress: (encodedFrames: Int) -> Unit = {}
    ): File {
        output.parentFile?.mkdirs()
        if (output.exists()) output.delete()

        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
            setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2)
            setInteger(MediaFormat.KEY_BITRATE_MODE, MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR)
        }

        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        var muxer: MediaMuxer? = null
        var muxerStarted = false
        var trackIndex = -1
        val info = MediaCodec.BufferInfo()
        var submitted = 0
        var completed = false

        try {
            muxer = MediaMuxer(output.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            codec.start()

            fun drain(endOfStream: Boolean) {
                var idleCount = 0
                while (true) {
                    val index = codec.dequeueOutputBuffer(info, if (endOfStream) 10_000 else 0)
                    when {
                        index == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                            if (!endOfStream || ++idleCount > 100) return
                        }
                        index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            check(!muxerStarted) { "Encoder output format changed more than once" }
                            trackIndex = muxer!!.addTrack(codec.outputFormat)
                            muxer.start()
                            muxerStarted = true
                        }
                        index >= 0 -> {
                            val encoded = codec.getOutputBuffer(index)
                            if (encoded != null && info.size > 0 && muxerStarted &&
                                info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0
                            ) {
                                encoded.position(info.offset)
                                encoded.limit(info.offset + info.size)
                                muxer!!.writeSampleData(trackIndex, encoded, info)
                            }
                            val eos = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                            codec.releaseOutputBuffer(index, false)
                            if (eos) {
                                completed = true
                                return
                            }
                        }
                    }
                }
            }

            produceFrames { source ->
                var inputIndex = codec.dequeueInputBuffer(10_000)
                while (inputIndex < 0) {
                    drain(false)
                    inputIndex = codec.dequeueInputBuffer(10_000)
                }
                val image = codec.getInputImage(inputIndex)
                    ?: error("This H.264 encoder did not expose a flexible YUV input image")
                try {
                    writeBitmapToYuv420(source, image)
                } finally {
                    image.close()
                }
                val ptsUs = submitted.toLong() * 1_000_000L / fps
                codec.queueInputBuffer(inputIndex, 0, width * height * 3 / 2, ptsUs, 0)
                submitted++
                drain(false)
                onProgress(submitted)
            }

            check(submitted >= 2) { "At least two frames are required for an MP4 video" }
            val eosIndex = codec.dequeueInputBuffer(10_000)
            check(eosIndex >= 0) { "Timed out while finishing the video encoder" }
            codec.queueInputBuffer(eosIndex, 0, 0, submitted.toLong() * 1_000_000L / fps,
                MediaCodec.BUFFER_FLAG_END_OF_STREAM)
            drain(true)
            check(completed && muxerStarted) { "Video encoder did not finish the MP4 stream" }
            check(output.isFile && output.length() > 10_000L) { "MP4 output is missing or too small" }
            return output
        } finally {
            try { codec.stop() } catch (_: Exception) {}
            codec.release()
            if (muxerStarted) {
                try { muxer?.stop() } catch (_: Exception) {}
            }
            muxer?.release()
        }
    }

    private fun writeBitmapToYuv420(bitmap: Bitmap, image: android.media.Image) {
        require(image.width == width && image.height == height) {
            "Encoder input image has unexpected dimensions: ${image.width}x${image.height}"
        }
        val source = if (bitmap.width == width && bitmap.height == height) bitmap
            else Bitmap.createScaledBitmap(bitmap, width, height, true)
        val owned = source !== bitmap
        try {
            val pixels = IntArray(width * height)
            source.getPixels(pixels, 0, width, 0, 0, width, height)
            val planes = image.planes
            require(planes.size == 3) { "Expected three YUV420 planes, got ${planes.size}" }

            val yPlane = planes[0]
            val uPlane = planes[1]
            val vPlane = planes[2]
            val yBuffer = yPlane.buffer
            val uBuffer = uPlane.buffer
            val vBuffer = vPlane.buffer
            yBuffer.clear(); uBuffer.clear(); vBuffer.clear()

            for (row in 0 until height) {
                val rowOffset = row * yPlane.rowStride
                for (col in 0 until width) {
                    val rgb = pixels[row * width + col]
                    val r = (rgb shr 16) and 0xff
                    val g = (rgb shr 8) and 0xff
                    val b = rgb and 0xff
                    val y = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
                    yBuffer.put(rowOffset + col * yPlane.pixelStride, y.coerceIn(0, 255).toByte())
                }
            }

            for (row in 0 until height / 2) {
                for (col in 0 until width / 2) {
                    var rSum = 0
                    var gSum = 0
                    var bSum = 0
                    for (dy in 0..1) for (dx in 0..1) {
                        val rgb = pixels[(row * 2 + dy) * width + col * 2 + dx]
                        rSum += (rgb shr 16) and 0xff
                        gSum += (rgb shr 8) and 0xff
                        bSum += rgb and 0xff
                    }
                    val r = rSum / 4
                    val g = gSum / 4
                    val b = bSum / 4
                    val u = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
                    val v = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
                    val uOffset = row * uPlane.rowStride + col * uPlane.pixelStride
                    val vOffset = row * vPlane.rowStride + col * vPlane.pixelStride
                    uBuffer.put(uOffset, u.coerceIn(0, 255).toByte())
                    vBuffer.put(vOffset, v.coerceIn(0, 255).toByte())
                }
            }
        } finally {
            if (owned) source.recycle()
        }
    }
}
