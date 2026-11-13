package com.romangold.tank2aivideostudio

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
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

@RunWith(AndroidJUnit4::class)
class RealImageGenerationInstrumentedTest {
    @Test(timeout = 5 * 60 * 60 * 1000L)
    fun realPhotoProducesPlayableMp4() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val outputDirectory = requireNotNull(context.getExternalFilesDir(null))
        outputDirectory.mkdirs()

        val sourceImage = File(outputDirectory, "ci-source.png")
        instrumentation.context.assets.open("ci-source.png").use { input ->
            sourceImage.outputStream().buffered().use { output -> input.copyTo(output) }
        }
        assertTrue("Real source photograph was not copied", sourceImage.isFile && sourceImage.length() > 10_000L)
        Log.i("RealImageI2V", "SOURCE_IMAGE=${sourceImage.absolutePath} bytes=${sourceImage.length()}")

        val totalStarted = SystemClock.elapsedRealtime()
        val modelFiles = MobileI2VModelManager(context).ensureModels { stage, copied, total ->
            Log.i("RealImageI2V", "MODEL $stage $copied/$total")
        }
        Log.i("RealImageI2V", "MODEL_PREP_MS=${SystemClock.elapsedRealtime() - totalStarted}")

        val generationStarted = SystemClock.elapsedRealtime()
        val engine = MobileI2VOnnxEngine(modelFiles) { stage, percent ->
            Log.i("RealImageI2V", "GENERATION $percent% $stage")
        }
        val sourceFrames = try {
            engine.generate(context.contentResolver, Uri.fromFile(sourceImage))
        } finally {
            engine.close()
        }

        try {
            assertEquals("MobileI2V must produce 17 generated source frames", 17, sourceFrames.size)
            Log.i("RealImageI2V", "MOBILEI2V_GENERATION_MS=${SystemClock.elapsedRealtime() - generationStarted}")

            val generatedFramePreview = File(outputDirectory, "ci-generated-frame.jpg")
            FileOutputStream(generatedFramePreview).use { stream ->
                assertTrue("Could not save generated frame preview",
                    sourceFrames.first().compress(Bitmap.CompressFormat.JPEG, 90, stream))
            }

            val outputVideo = File(outputDirectory, "ci-generated.mp4")
            val rifeStarted = SystemClock.elapsedRealtime()
            val rife = RifeX2Interpolator(context)
            try {
                VideoMp4Encoder(outputVideo, fps = 16).encodeStream(
                    produceFrames = { emit ->
                        rife.forEachInterpolatedFrame(
                            sourceFrames,
                            onFrameProgress = { done, total ->
                                Log.i("RealImageI2V", "RIFE_PROGRESS=$done/$total")
                            },
                            onFrame = { frame, _ -> emit(frame) }
                        )
                    },
                    onProgress = { count ->
                        Log.i("RealImageI2V", "MP4_ENCODED_FRAMES=$count")
                    }
                )
            } finally {
                rife.close()
            }
            Log.i("RealImageI2V", "RIFE_AND_MP4_MS=${SystemClock.elapsedRealtime() - rifeStarted}")

            assertTrue("Generated MP4 is missing or too small",
                outputVideo.isFile && outputVideo.length() > 10_000L)

            val metadata = MediaMetadataRetriever()
            try {
                metadata.setDataSource(outputVideo.absolutePath)
                val durationMs = metadata.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                    ?.toLongOrNull() ?: 0L
                val width = metadata.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)
                    ?.toIntOrNull() ?: 0
                val height = metadata.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
                    ?.toIntOrNull() ?: 0
                assertTrue("Generated MP4 duration is invalid: $durationMs ms", durationMs >= 1_500L)
                assertEquals("Generated MP4 width", 1280, width)
                assertEquals("Generated MP4 height", 720, height)

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
                Log.i("RealImageI2V", summary.replace("\n", " | "))
            } finally {
                metadata.release()
            }
        } finally {
            sourceFrames.forEach { if (!it.isRecycled) it.recycle() }
        }
    }
}
