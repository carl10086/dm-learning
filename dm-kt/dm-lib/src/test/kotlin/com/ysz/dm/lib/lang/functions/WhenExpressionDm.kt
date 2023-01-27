package com.ysz.dm.lib.lang.functions

import com.ysz.dm.lib.common.atomictest.eq
import com.ysz.dm.lib.common.atomictest.trace
import org.junit.jupiter.api.Test

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2023/1/12
 **/
internal class WhenExpressionDm {

    private fun processInputs(inputs: List<String>) {
        val coordinates = Coordinates()
        for (input in inputs) {
            when (input) {
                "up", "u" -> coordinates.y--
                "down", "d" -> coordinates.y++
                "left", "l" -> coordinates.x--
                "right", "r" -> {
                    trace("Moving right")
                    coordinates.x++
                }

                "nowhere" -> {}
                "exit" -> return
                else -> trace("bad input :$input")
            }
        }
    }

    @Test
    fun `test move`() {
        processInputs(
            listOf(
                "up", "d", "nowhere", "left", "right", "exit", "r"
            )
        )
        trace eq """
            y gets -1
            y gets 0
            x gets -1
            Moving right
            x gets 0
        """.trimIndent()
    }

    @Test
    fun `test trace`() {
        Coordinates().apply {
            x = 1
            y = 2
        }

        trace eq """
            x gets 1
            y gets 2
        """.trimIndent()
    }

    private fun mixColors(first: String, second: String) =
        /*you can use expressions here*/
        when (setOf(first, second)) {
            setOf("red", "blue") -> "purple"
            setOf("red", "yellow") -> "orange"
            setOf("blue", "yellow") -> "green"
            else -> "unknown"
        }

    @Test
    fun `test expression`() {
        mixColors("red", "blue") eq "purple"
    }

    private fun bmiMetricOld(kg: Double, height: Double): String {
        val bmi = kg / (height * height)

        return if (bmi < 18.5) "Underweight"
        else if (bmi < 25) "Normal weight"
        else "Overweight"
    }

    private fun bmiMetricWithWhen(kg: Double, height: Double): String {
        val bmi = kg / (height * height)
        return when {
            bmi < 18.5 -> "Underweight"
            bmi < 25 -> "Normal weight"
            else -> "Overweight"
        }
    }


    @Test
    fun `test noArgs`() {
        bmiMetricOld(72.57, 1.727) eq bmiMetricWithWhen(72.57, 1.727)
    }
}


class Coordinates {
    override fun toString(): String {
        return "($x, $y)"
    }

    var x: Int = 0
        set(value) {
            trace("x gets $value")
            field = value
        }

    var y: Int = 0
        set(value) {
            trace("y gets $value")
            field = value
        }

}