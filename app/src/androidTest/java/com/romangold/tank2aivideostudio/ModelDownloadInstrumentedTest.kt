package com.romangold.tank2aivideostudio

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class ModelDownloadInstrumentedTest {
    @Test
    fun mainActivityLaunchesOnAndroidRuntime() {
        ActivityScenario.launch(MainActivity::class.java).use {
            assertTrue("MainActivity must reach RESUMED state", it.state.name == "RESUMED")
        }
    }

    @Test
    fun extractsAllBundledModelsAndRifeWithoutNetwork() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val started = SystemClock.elapsedRealtime()
        val files = MobileI2VModelManager(context).ensureModels { stage, copied, total ->
            Log.i("Tank2EmbeddedTest", "$stage: $copied/$total")
        }
        val models = listOf(files.encoder, files.denoiser, files.decoder)
        models.forEach { file ->
            assertTrue("Missing or empty model: ${file.name}", file.isFile && file.length() > 1_000_000L)
            Log.i("Tank2EmbeddedTest", "VERIFIED ${file.name} ${file.length()} bytes")
        }
        val rifeProbe = File(context.cacheDir, "rife-asset-probe.onnx")
        context.assets.open("models/rife49.onnx").use { input ->
            rifeProbe.outputStream().use { output -> input.copyTo(output) }
        }
        assertTrue("Bundled RIFE asset is missing", rifeProbe.length() > 1_000_000L)
        rifeProbe.delete()
        assertTrue("All models must be stored in app-private storage",
            models.all { it.canonicalPath.startsWith(context.filesDir.canonicalPath) })
        Log.i("Tank2EmbeddedTest", "MODEL_EXTRACTION_MS=${SystemClock.elapsedRealtime() - started}")
    }

    @Test
    fun generatesVideoLocallyAndExportsMp4OnAndroidRuntime() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val started = SystemClock.elapsedRealtime()
        val modelFiles = MobileI2VModelManager(context).ensureModels()
        val inputBitmap = Bitmap.createBitmap(1280, 720, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(inputBitmap)
        canvas.drawColor(android.graphics.Color.rgb(28, 48, 72))
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = android.graphics.Color.rgb(245, 170, 65)
            style = Paint.Style.FILL
        }
        canvas.drawCircle(640f, 360f, 170f, paint)
        val inputFile = File(context.cacheDir, "tank2-inference-input.jpg")
        inputFile.outputStream().use { inputBitmap.compress(Bitmap.CompressFormat.JPEG, 92, it) }
        inputBitmap.recycle()

        var sourceFrames: List<Bitmap> = emptyList()
        try {
            Log.i("Tank2EmbeddedTest", "ANDROID_INFERENCE_START")
            val engine = MobileI2VOnnxEngine(modelFiles) { stage, pct ->
                Log.i("Tank2EmbeddedTest", "INFERENCE $stage $pct%")
            }
            sourceFrames = try {
                engine.generate(context.contentResolver, Uri.fromFile(inputFile))
            } finally {
                engine.close()
            }
            assertTrue("Expected 17 decoded frames, got ${sourceFrames.size}", sourceFrames.size == 17)
            Log.i("Tank2EmbeddedTest", "INFERENCE_DONE_MS=${SystemClock.elapsedRealtime() - started}")

            val output = File(context.filesDir, "generated/android-emulator-test.mp4")
            val rife = RifeX2Interpolator(context)
            try {
                VideoMp4Encoder(output, fps = 16).encodeStream(
                    produceFrames = { emit ->
                        rife.forEachInterpolatedFrame(
                            sourceFrames,
                            onModelDownloadProgress = { done, total ->
                                Log.i("Tank2EmbeddedTest", "RIFE_ASSET_COPY $done/$total")
                            },
                            onFrameProgress = { done, total ->
                                Log.i("Tank2EmbeddedTest", "RIFE $done/$total")
                            },
                            onFrame = { frame, _ -> emit(frame) }
                        )
                    },
                    onProgress = { count ->
                        if (count % 4 == 0) Log.i("Tank2EmbeddedTest", "MP4_ENCODED_FRAMES=$count")
                    }
                )
            } finally {
                rife.close()
            }
            assertTrue("MP4 output is missing or too small", output.isFile && output.length() > 10_000L)
            val metadata = MediaMetadataRetriever()
            try {
                metadata.setDataSource(output.absolutePath)
                val duration = metadata.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
                assertTrue("MP4 duration is invalid: $duration ms", duration >= 1_000L)
                Log.i("Tank2EmbeddedTest", "MP4_BYTES=${output.length()}")
                Log.i("Tank2EmbeddedTest", "MP4_DURATION_MS=$duration")
            } finally {
                metadata.release()
            }
            Log.i("Tank2EmbeddedTest", "FULL_GENERATION_SUCCESS_MS=${SystemClock.elapsedRealtime() - started}")
        } finally {
            sourceFrames.forEach { if (!it.isRecycled) it.recycle() }
            inputFile.delete()
        }
    }
}
