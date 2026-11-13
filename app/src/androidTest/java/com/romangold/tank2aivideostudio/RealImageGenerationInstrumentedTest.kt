package com.romangold.tank2aivideostudio

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicReference

@RunWith(AndroidJUnit4::class)
class RealImageGenerationInstrumentedTest {
    @Test(timeout = 5 * 60 * 60 * 1000L)
    fun realPhotoProducesPlayableMp4() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val outputDirectory = requireNotNull(context.getExternalFilesDir(null))
        outputDirectory.mkdirs()
        val totalStarted = SystemClock.elapsedRealtime()
        val currentStage = AtomicReference("TEST_START")
        val heartbeat = Thread({
            while (!Thread.currentThread().isInterrupted) {
                try {
                    Thread.sleep(30_000L)
                    logSnapshot("HEARTBEAT", currentStage.get(), totalStarted, outputDirectory)
                } catch (_: InterruptedException) {
                    return@Thread
                } catch (error: Throwable) {
                    Log.w(TAG, "Heartbeat diagnostic failed", error)
                }
            }
        }, "i2v-diagnostic-heartbeat").apply {
            isDaemon = true
            start()
        }

        fun stage(name: String, detail: String = "") {
            currentStage.set(name)
            logSnapshot("STAGE", "$name $detail".trim(), totalStarted, outputDirectory)
        }

        try {
            stage("COPY_SOURCE_IMAGE")
            val sourceImage = File(outputDirectory, "ci-source.png")
            instrumentation.context.assets.open("ci-source.png").use { input ->
                sourceImage.outputStream().buffered().use { output -> input.copyTo(output) }
            }
            assertTrue("Real source photograph was not copied", sourceImage.isFile && sourceImage.length() > 10_000L)
            Log.i(TAG, "SOURCE_IMAGE path=${sourceImage.absolutePath} bytes=${sourceImage.length()}")

            stage("INSTALL_AND_VERIFY_MODELS")
            val modelStarted = SystemClock.elapsedRealtime()
            val modelLastPercent = HashMap<String, Int>()
            val modelFiles = MobileI2VModelManager(context).ensureModels { name, copied, total ->
                val percent = if (total > 0) ((copied * 100L) / total).toInt() else -1
                val previous = modelLastPercent[name] ?: -5
                if (percent < 0 || percent == 100 || percent >= previous + 5) {
                    modelLastPercent[name] = percent
                    Log.i(TAG, "MODEL_COPY name=$name bytes=$copied/$total percent=$percent")
                }
            }
            Log.i(TAG, "MODEL_INSTALL_DONE elapsedMs=${SystemClock.elapsedRealtime() - modelStarted}")
            listOf(modelFiles.encoder, modelFiles.denoiser, modelFiles.decoder).forEach {
                Log.i(TAG, "MODEL_READY name=${it.name} bytes=${it.length()} path=${it.absolutePath}")
            }
            stage("MODELS_READY")

            stage("MOBILEI2V_BEGIN")
            val generationStarted = SystemClock.elapsedRealtime()
            val engine = MobileI2VOnnxEngine(modelFiles) { name, percent ->
                Log.i(TAG, "I2V_PROGRESS percent=$percent stage=$name elapsedMs=${SystemClock.elapsedRealtime() - generationStarted}")
            }
            val sourceFrames = try {
                engine.generate(context.contentResolver, Uri.fromFile(sourceImage))
            } finally {
                engine.close()
                Log.i(TAG, "I2V_ENGINE_CLOSED")
            }

            try {
                Log.i(TAG, "I2V_DONE frames=${sourceFrames.size} elapsedMs=${SystemClock.elapsedRealtime() - generationStarted}")
                assertEquals("MobileI2V must produce 17 generated source frames", 17, sourceFrames.size)
                stage("SAVE_GENERATED_FRAME_PREVIEW")
                val generatedFramePreview = File(outputDirectory, "ci-generated-frame.jpg")
                FileOutputStream(generatedFramePreview).use { stream ->
                    assertTrue("Could not save generated frame preview",
                        sourceFrames.first().compress(Bitmap.CompressFormat.JPEG, 90, stream))
                }
                Log.i(TAG, "PREVIEW_SAVED path=${generatedFramePreview.absolutePath} bytes=${generatedFramePreview.length()}")

                stage("RIFE_AND_MP4_BEGIN", "sourceFrames=${sourceFrames.size}")
                val outputVideo = File(outputDirectory, "ci-generated.mp4")
                val rifeStarted = SystemClock.elapsedRealtime()
                val rife = RifeX2Interpolator(context)
                try {
                    VideoMp4Encoder(outputVideo, fps = 16).encodeStream(
                        produceFrames = { emit ->
                            Log.i(TAG, "RIFE_BEGIN adjacentPairs=${sourceFrames.size - 1}")
                            rife.forEachInterpolatedFrame(
                                sourceFrames,
                                onModelDownloadProgress = { copied, total ->
                                    Log.i(TAG, "RIFE_MODEL_COPY bytes=$copied total=$total")
                                },
                                onFrameProgress = { done, total ->
                                    Log.i(TAG, "RIFE_PAIR_PROGRESS pair=$done/$total elapsedMs=${SystemClock.elapsedRealtime() - rifeStarted}")
                                },
                                onFrame = { frame, generated ->
                                    emit(frame)
                                    if (!generated) {
                                        Log.i(TAG, "FRAME_SUBMITTED type=source size=${frame.width}x${frame.height}")
                                    }
                                }
                            )
                            Log.i(TAG, "RIFE_DONE elapsedMs=${SystemClock.elapsedRealtime() - rifeStarted}")
                        },
                        onProgress = { count ->
                            Log.i(TAG, "MP4_ENCODE_PROGRESS frames=$count bytes=${outputVideo.length()} elapsedMs=${SystemClock.elapsedRealtime() - rifeStarted}")
                        }
                    )
                } finally {
                    rife.close()
                    Log.i(TAG, "RIFE_ENGINE_CLOSED")
                }
                Log.i(TAG, "RIFE_AND_MP4_DONE elapsedMs=${SystemClock.elapsedRealtime() - rifeStarted} bytes=${outputVideo.length()}")
                assertTrue("Generated MP4 is missing or too small",
                    outputVideo.isFile && outputVideo.length() > 10_000L)

                stage("VERIFY_MP4_METADATA")
                val metadata = MediaMetadataRetriever()
                try {
                    metadata.setDataSource(outputVideo.absolutePath)
                    val durationMs = metadata.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                        ?.toLongOrNull() ?: 0L
                    val width = metadata.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)
                        ?.toIntOrNull() ?: 0
                    val height = metadata.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
                        ?.toIntOrNull() ?: 0
                    Log.i(TAG, "MP4_METADATA durationMs=$durationMs width=$width height=$height bytes=${outputVideo.length()}")
                    assertTrue("Generated MP4 duration is invalid: $durationMs ms", durationMs >= 1_500L)
                    assertEquals("Generated MP4 width", 1280, width)
                    assertEquals("Generated MP4 height", 720, height)

                    stage("DECODE_MP4_PREVIEW_FRAME")
                    val preview = metadata.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                    assertNotNull("Generated MP4 could not be decoded", preview)
                    preview?.let { bitmap ->
                        try {
                            FileOutputStream(File(outputDirectory, "ci-video-preview.jpg")).use { stream ->
                                assertTrue("Could not save decoded MP4 preview",
                                    bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream))
                            }
                        } finally {
                            bitmap.recycle()
                        }
                    }

                    val summary = listOf(
                        "RESULT=SUCCESS",
                        "SOURCE_IMAGE=ci-source.png",
                        "OUTPUT_VIDEO=ci-generated.mp4",
                        "VIDEO_BYTES=${outputVideo.length()}",
                        "VIDEO_DURATION_MS=$durationMs",
                        "VIDEO_WIDTH=$width",
                        "VIDEO_HEIGHT=$height",
                        "SOURCE_FRAMES=${sourceFrames.size}",
                        "OUTPUT_FPS=16",
                        "TOTAL_ELAPSED_MS=${SystemClock.elapsedRealtime() - totalStarted}"
                    ).joinToString("\n") + "\n"
                    File(outputDirectory, "ci-generation-summary.txt").writeText(summary)
                    Log.i(TAG, summary.replace("\n", " | "))
                    stage("TEST_SUCCESS")
                } finally {
                    metadata.release()
                }
            } finally {
                sourceFrames.forEach { if (!it.isRecycled) it.recycle() }
                Log.i(TAG, "SOURCE_FRAMES_RELEASED")
            }
        } catch (error: Throwable) {
            currentStage.set("FAILED")
            Log.e(TAG, "TEST_FAILED stage=FAILED elapsedMs=${SystemClock.elapsedRealtime() - totalStarted} message=${error.message}", error)
            throw error
        } finally {
            heartbeat.interrupt()
            Log.i(TAG, "TEST_FINALLY elapsedMs=${SystemClock.elapsedRealtime() - totalStarted}")
        }
    }

    private fun logSnapshot(kind: String, stage: String, startedAt: Long, directory: File) {
        val runtime = Runtime.getRuntime()
        val usedHeap = runtime.totalMemory() - runtime.freeMemory()
        val freeDisk = directory.usableSpace
        Log.i(
            TAG,
            "$kind stage=$stage elapsedMs=${SystemClock.elapsedRealtime() - startedAt} " +
                "heapUsedBytes=$usedHeap heapCommittedBytes=${runtime.totalMemory()} heapMaxBytes=${runtime.maxMemory()} " +
                "nativeHeapBytes=${Debug.getNativeHeapAllocatedSize()} diskFreeBytes=$freeDisk"
        )
    }

    companion object {
        private const val TAG = "RealImageI2V"
    }
}
