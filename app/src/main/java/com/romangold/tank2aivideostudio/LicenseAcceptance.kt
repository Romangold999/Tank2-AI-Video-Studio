package com.romangold.tank2aivideostudio

import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.text.SpannableString
import android.text.Spanned
import android.text.method.LinkMovementMethod
import android.text.style.ClickableSpan
import android.view.View
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/** Enforceable acceptance gate for the LTX Video Open Weights License. */
object LicenseAcceptance {
    private const val PREFS = "model_license_acceptance"
    private const val KEY_ACCEPTED = "ltx_video_open_weights_0_x_accepted"
    private const val LICENSE_URL = "https://static.lightricks.com/legal/LTX-Video-Open-Weights-License-0.X.pdf"

    fun requireAccepted(activity: Activity, onAccepted: () -> Unit) {
        val prefs = activity.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        if (prefs.getBoolean(KEY_ACCEPTED, false)) {
            onAccepted()
            return
        }

        val layout = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 24, 48, 8)
        }
        val scroll = ScrollView(activity)
        val body = TextView(activity).apply {
            setTextColor(Color.DKGRAY)
            textSize = 15f
            text = SpannableString(
                "Для загрузки и использования LTX VAE необходимо принять LTX Video Open Weights License 0.X.\n\n" +
                "Краткое изложение ограничений (полный текст имеет приоритет):\n" +
                "• Использовать модель и производные материалы только в соответствии с лицензией и законом.\n" +
                "• Не использовать для незаконной деятельности, причинения вреда несовершеннолетним, преследования, вредоносной дезинформации или нарушения прав других людей.\n" +
                "• Не создавать несанкционированные имитации личности или интимные изображения без согласия.\n" +
                "• Не применять в запрещённых лицензией дискриминационных и иных ограниченных сценариях.\n\n" +
                "Полная копия лицензии включена в APK: assets/licenses/LTX-Video-Open-Weights-License-0.X.pdf. Ниже доступен официальный оригинал. Это краткое изложение не заменяет полный текст."
            )
            movementMethod = LinkMovementMethod.getInstance()
        }
        val base = body.text.toString()
        val label = "Открыть полный текст лицензии (PDF)"
        val linked = SpannableString("$base\n\n$label")
        linked.setSpan(object : ClickableSpan() {
            override fun onClick(widget: View) {
                try {
                    activity.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(LICENSE_URL)))
                } catch (_: Exception) { }
            }
        }, base.length + 2, linked.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        body.text = linked
        scroll.addView(body)
        layout.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))

        val check = CheckBox(activity).apply {
            text = "Я прочитал(а) и принимаю полные условия лицензии"
            setTextColor(Color.DKGRAY)
            textSize = 14f
        }
        layout.addView(check)

        val dialog = AlertDialog.Builder(activity)
            .setTitle("Условия использования модели")
            .setView(layout)
            .setNegativeButton("Отмена", null)
            .setPositiveButton("Принять и продолжить", null)
            .create()
        dialog.setOnShowListener {
            val accept = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
            accept.isEnabled = false
            check.setOnCheckedChangeListener { _, checked -> accept.isEnabled = checked }
            accept.setOnClickListener {
                prefs.edit().putBoolean(KEY_ACCEPTED, true).apply()
                dialog.dismiss()
                onAccepted()
            }
        }
        dialog.setCancelable(false)
        dialog.show()
    }
}
