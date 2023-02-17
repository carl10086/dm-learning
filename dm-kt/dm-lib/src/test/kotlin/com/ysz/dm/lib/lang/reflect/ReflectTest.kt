package com.ysz.dm.lib.lang.reflect

import com.ysz.dm.lib.java.JavaObj
import org.junit.jupiter.api.Test


/**
 * @author carl
 * @since 2023-02-17 9:30 PM
 **/
internal class ReflectTest {
    @Test
    fun `test  kotlin field`() {
        val kClazz = ReflectDataClz::class

        val constructors = kClazz.constructors
        val constructor = constructors.first()
        println("finish")
    }

    @Test
    fun `test java field`() {
        val clz = JavaObj::class.java
        val kClass = clz.kotlin
        val constructor = kClass.constructors.first()
        println("finish")

    }
}


data class ReflectDataClz(
    val id: Long,
    @field:Transient
    val ignore: String?,
    val username: String
)