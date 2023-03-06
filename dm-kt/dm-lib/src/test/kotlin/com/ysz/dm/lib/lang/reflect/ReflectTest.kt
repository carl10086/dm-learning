package com.ysz.dm.lib.lang.reflect

import com.ysz.dm.lib.java.JavaObj
import com.ysz.dm.lib.java.JavaRecord
import org.junit.jupiter.api.Test
import kotlin.reflect.full.memberProperties


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
        val ins1 = constructor.call(100L, null, "carl")
        println(ins1)
        val memberProperties = kClazz.memberProperties
        for (memberProperty in memberProperties) {
            println(memberProperty.get(ins1))
        }
    }

    @Test
    fun `test java field`() {
        val clz = JavaObj::class.java
        val kClass = clz.kotlin
        val constructor = kClass.constructors.first()
        println("finish")
    }

    @Test
    fun `test set Method`() {
        val klass = ReflectDataClz::class
        klass.members.forEach {
            println(it)
        }
    }

    @Test
    fun `test withIndex`() {
        val counts = listOf(1, 2, 3)
        counts.withIndex().forEach { println(it) }
    }

    @Test
    fun `test iterator`() {
        val a1 = listOf(1L, 2L)
        println(a1 is Collection<Any>)
    }

    @Test
    fun `test record`() {
        val kclass = JavaRecord::class
        println(kclass.isData)
    }

    @Test
    fun `test NullAble`() {
        val prop = ReflectDataClz::class.memberProperties.first { it.name == "id" }
        println(prop)
    }
}


data class ReflectDataClz(
    var id: Long?,
    @field:Transient
    val ignore: String?,
    val username: String,
)