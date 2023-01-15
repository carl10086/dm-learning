package com.ysz.dm.lib.lang.functions

import com.ysz.dm.lib.common.atomictest.eq
import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-01-15 1:53 PM
 **/
internal class ManipulateDm {

    @Test
    fun `test zip`() {
        val left = listOf("a", "b", "c", "d")
        val right = listOf("q", "r", "s", "t")

        left.zip(right).toString() eq "[(a, q), (b, r), (c, s), (d, t)]"
        left.zip(0..4) eq "[(a, 0), (b, 1), (c, 2), (d, 3)]"
        (10..100).zip(right) eq "[(10, q), (11, r), (12, s), (13, t)]"
    }


    @Test
    fun `test zipOnPair`() {
        val names = listOf("Bob", "Jill", "Jim")
        val ids = listOf(1731, 9274, 8378)


        fun Person.greatThan(): Boolean {
            return this.id > 8378
        }

        names.zip(ids, ::Person).filter(Person::greatThan) eq "[Person(name=Jill, id=9274)]"
    }


    @Test
    fun `test BasicFlatten`() {
        val list = listOf(
            listOf(1, 2), listOf(4, 5)
        )

        list.flatten() eq "[1, 2, 4, 5]"
    }


    @Test
    fun `test baseFlatMap`() {
        val intRange = 1..3

        // List<List<Pair<Int, Int>>>
        intRange.map { a ->
            intRange.map { b -> a to b }
        } eq "[[(1, 1), (1, 2), (1, 3)], [(2, 1), (2, 2), (2, 3)], [(3, 1), (3, 2), (3, 3)]]"

        intRange.map { a -> intRange.map { b -> a to b } }
            .flatten() eq "[(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]"

        intRange.flatMap { a ->
            intRange.map { b -> a to b }
        } eq "[(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]"
    }


    @Test
    fun `test flatMapWithBook`() {
        val books = listOf(
            Book("1984", listOf("George Orwell")),
            Book("Ulysses", listOf("James Joyce"))
        )

        books.flatMap { it.authors } eq "[George Orwell, James Joyce]"
        books.map { it.authors }.flatten() eq "[George Orwell, James Joyce]"
    }
}

data class Person(
    val name: String, val id: Int
)

data class Book(
    val title: String, val authors: List<String>
)