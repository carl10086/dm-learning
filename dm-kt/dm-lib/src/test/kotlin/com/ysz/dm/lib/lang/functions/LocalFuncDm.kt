package com.ysz.dm.lib.lang.functions

import com.ysz.dm.lib.common.atomictest.eq
import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-01-16 12:22 AM
 **/
internal class LocalFuncDm {
    @Test
    fun `test local`() {
        val logMsg = StringBuilder()
        fun log(message: String) = logMsg.appendLine(message)

        log("Starting computation")
        val x = 42
        log("computation result: $x")

        logMsg.toString() eq """
            Starting computation
            computation result: 42
        """.trimIndent()
    }


    data class Session(
        val title: String,
        val speaker: String
    )

    private val sessions = listOf(Session("Kotlin", "Roman Elizarov"))
    private val favSpeakers = setOf("Roman Elizarov")

    @Test
    fun `test local lambda`() {
        fun interesting(session: Session): Boolean =
            session.title.contains("Kotlin") && favSpeakers.contains(session.speaker)

        sessions.any(::interesting) eq true
    }

    @Test
    fun `test local anonymous`() {
        sessions.any(fun(session: Session): Boolean {
            return session.title.contains("Kotlin") && favSpeakers.contains(session.speaker)
        }) eq true
    }
}