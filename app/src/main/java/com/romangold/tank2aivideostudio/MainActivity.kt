package com.romangold.tank2aivideostudio

import android.app.Activity
import android.os.Bundle
import android.graphics.Color
import android.view.Gravity
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

class MainActivity : Activity() {
    private val bg = Color.rgb(16, 20, 28)
    private val panel = Color.rgb(29, 36, 48)
    private val muted = Color.rgb(174, 185, 202)
    private val accent = Color.rgb(116, 166, 255)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(22, 28, 22, 24)
            setBackgroundColor(bg)
        }
        val scroll = ScrollView(this)
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 0, 0, 24)
        }
        content.addView(text("TANK2 AI VIDEO STUDIO", 23f, Color.WHITE, true))
        content.addView(text("Offline-first • Android • Helio G99", 14f, muted, false).withTop(8))
        content.addView(card(
            "IMAGE → VIDEO",
            "Цель: настоящая генерация движения нейросетевой моделью. Модель и локальный runtime пока не подключены.",
            accent
        ).withTop(24))
        content.addView(card(
            "TEXT → VIDEO",
            "Отдельное направление исследований. На этом этапе генерация по тексту не реализована.",
            accent
        ).withTop(14))
        content.addView(card(
            "УСТРОЙСТВО",
            "Android ${android.os.Build.VERSION.RELEASE} • ${android.os.Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown"}\n" +
                "Модель устройства: ${android.os.Build.MODEL}\n" +
                "Ускорение нейросетей ещё не проверено.",
            muted
        ).withTop(14))
        content.addView(text(
            "ВАЖНО: приложение пока является стартовой оболочкой, а не генератором видео. " +
                "Мы не будем выдавать слайд-шоу или эффекты панорамирования за AI-видеогенерацию.",
            13f, muted, false
        ).withTop(20))
        scroll.addView(content)
        root.addView(scroll, ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))
        setContentView(root)
    }

    private fun text(value: String, size: Float, color: Int, bold: Boolean): TextView =
        TextView(this).apply {
            text = value
            textSize = size
            setTextColor(color)
            if (bold) setTypeface(typeface, android.graphics.Typeface.BOLD)
            setLineSpacing(4f, 1f)
        }

    private fun card(title: String, body: String, marker: Int): LinearLayout =
        LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(18, 18, 18, 18)
            setBackgroundColor(panel)
            addView(text(title, 16f, marker, true))
            addView(text(body, 14f, Color.WHITE, false).withTop(10))
        }

    private fun TextView.withTop(dp: Int): TextView {
        val px = (dp * resources.displayMetrics.density).toInt()
        layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { topMargin = px }
        return this
    }

    private fun LinearLayout.withTop(dp: Int): LinearLayout {
        val px = (dp * resources.displayMetrics.density).toInt()
        layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { topMargin = px }
        return this
    }
}
