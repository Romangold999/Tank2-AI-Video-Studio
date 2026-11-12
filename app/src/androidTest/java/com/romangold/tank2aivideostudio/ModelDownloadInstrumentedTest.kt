package com.romangold.tank2aivideostudio

import android.util.Log
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ModelDownloadInstrumentedTest {
    @Test
    fun mainActivityLaunchesOnAndroidRuntime() {
        ActivityScenario.launch(MainActivity::class.java).use {
            assertTrue("MainActivity must reach RESUMED state", it.state.name == "RESUMED")
        }
    }

    @Test
    fun downloadsAndVerifiesAllPublishedModelsInsideAndroidAppStorage() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val files = MobileI2VModelManager(context).ensureModels { stage, downloaded, total ->
            Log.i("Tank2ModelTest", "$stage: $downloaded/$total")
        }
        val models = listOf(files.encoder, files.denoiser, files.decoder)
        models.forEach { file ->
            assertTrue("Missing or empty model: ${file.name}", file.isFile && file.length() > 1_000_000L)
            Log.i("Tank2ModelTest", "VERIFIED ${file.name} ${file.length()} bytes")
        }
        assertTrue("All models must be stored in app-private storage",
            models.all { it.canonicalPath.startsWith(context.filesDir.canonicalPath) })
    }
}
