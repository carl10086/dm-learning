package com.ysz.dm.base.core.tools.json

import com.fasterxml.jackson.module.kotlin.readValue
import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-01-06 9:48 PM
 */
internal class JsonToolsTest {

    @Test
    fun `test json`() {
        val jsonStr = JsonTools.mapper.writeValueAsString(Child("aaa", 10))
        println(jsonStr)
        println(JsonTools.mapper.readValue<Child>(jsonStr))
        println(JsonTools.mapper.readValue<Base>(jsonStr))

    }
}



data class Child(
    val name: String,
    val age: Int
) : Base(name)