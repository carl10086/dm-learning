package com.ysz.dm.lib.lang.collection

import com.ysz.dm.lib.common.atomictest.eq
import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-01-15 11:52 PM
 **/
internal class BuildMapDm {

    data class Person(
        val name: String,
        val age: Int
    )

    @Test
    fun `test build groupBy`() {
        val persons = people()
        val map = persons.groupBy(Person::age)
        map[20] eq "[Person(name=carl, age=20)]"
        map[21] eq null
    }

    private fun people(): List<Person> {
        val names = listOf("carl", "zoe")
        val ages = listOf(20, 10)
        return names.zip(ages, ::Person)
    }

    @Test
    fun `test build associateWith`() {
        val map = people().associateWith { it.name }

        map[Person("carl", 20)] eq "carl"
    }
}