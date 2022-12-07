package com.ysz.dm.lib.lang.collection

import org.junit.jupiter.api.Test

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/16
 **/
internal class AssociateMapTest {


    @Test
    internal fun `test_assocaite`() {
        data class Person(val name: String, val city: String, val phone: String) // 1

        val people = listOf(                                                     // 2
            Person("John", "Boston", "+1-888-123456"),
            Person("Sarah", "Munich", "+49-777-789123"),
            Person("Svyatoslav", "Saint-Petersburg", "+7-999-456789"),
            Person("Vasilisa", "Saint-Petersburg", "+7-999-123456"),
            Person("John", "testtest", "+7-999-123456")
        )

        val phoneBook = people.associateBy { it.phone }                          // 3
        val cityBook = people.associateBy(Person::phone, Person::city)
            .forEach { (t, u) -> println("k:$t, v:$u") }           // 4
        val peopleCities = people.groupBy(Person::city, Person::name)            // 5
        val lastPersonCity = people.associateBy(Person::city, Person::name)
    }


    @Test
    fun `test_containsKey`() {
        val map: Map<String, List<String>> = buildMap {
            put("a", listOf("a"))
        }

        map["b"]
            ?.
            let {
            println("out : $it")
            it.forEach { x ->
                println("why :$x")
            }
            for (s in it) {
                println("what :$s")
            }
        } ?: let {
            println("no data")
        }


        var b: String? = null

        b?.let { println("b is not null") } ?: let { println("b is null") }
    }
}