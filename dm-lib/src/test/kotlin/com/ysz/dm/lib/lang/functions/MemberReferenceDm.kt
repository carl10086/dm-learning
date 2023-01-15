package com.ysz.dm.lib.lang.functions

import com.ysz.dm.lib.common.atomictest.eq
import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-01-15 1:00 AM
 **/
internal class MemberReferenceDm {

    @Test
    fun `test propertyReference`() {
        val messages = listOf(
            Message("Kitty", "Hey!", true),
            Message("Kitty", "Where are you?", false),
            Message("Boss", "Meeting today", false)
        )

        messages.sortedWith(
            compareBy(
                Message::isRead,
                Message::sender
            )
        )

    }


    @Test
    fun `test constructorReference`() {
        val names = listOf("Alice", "Bob")
        val students = names.mapIndexed { index, name ->
            Student(index, name)
        }

        students eq listOf(Student(0, "Alice"), Student(1, "Bob"))

        /*this is the way use constructor reference*/
        names.mapIndexed(::Student) eq students


    }
}

data class Message(
    val sender: String,
    val text: String,
    val isRead: Boolean
)

data class Student(
    val id: Int,
    val name: String
)