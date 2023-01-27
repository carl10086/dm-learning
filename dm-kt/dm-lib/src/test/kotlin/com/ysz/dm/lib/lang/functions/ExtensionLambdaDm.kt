package com.ysz.dm.lib.lang.functions

import com.ysz.dm.lib.common.atomictest.eq
import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-01-16 11:21 AM
 **/
internal class ExtensionLambdaDm {

    @Test
    fun `test basic`() {
        val va: (String, Int) -> String = { str, n ->
            str.repeat(n) + str.repeat(n)
        }


        val vb: String.(Int) -> String = {
            this.repeat(it) + repeat(it)
        }


        va("Vanbo", 2).eq("VanboVanboVanboVanbo")
        vb("Vanbo", 2).eq("VanboVanboVanboVanbo")
        "Vanbo".vb(2).eq("VanboVanboVanboVanbo")
    }


    @Test
    fun `test multiple params`() {
        val zero: Int.() -> Boolean = {
            this == 0
        }

        val one: Int.(Int) -> Boolean = {
            this % it == 0
        }

        val two: Int.(Int, Int) -> Boolean = { arg1, arg2 ->
            (arg1 + arg2) == 0
        }

        0.zero() eq true
        10.one(10) eq true
        20.two(10, 10) eq true
    }


    @Test
    fun `test func references`() {
        fun Int.d1(f: (Int) -> Int) = f(this) * 10

        /*f 是一个 extension 函数 . 只是参数 变为 this*/
        fun Int.d2(f: Int.() -> Int) = f() * 10


        val f1: (Int) -> Int = {
            it + 3
        }

        fun Int.f2() = this + 3

        fun f3(n: Int) = n + 3


        74.d1 { f1(it) } eq 770
        74.d1(::f3) eq 770
        74.d1(Int::f2) eq 770


        74.d2(f1) eq 770

        74.d2(Int::f2) eq 770
        74.d2(::f3) eq 770
    }


    open class Base {
        open fun f() = 1
    }

    class Derived : Base() {
        override fun f() = 99
    }

    @Test
    fun `test polymorphism`() {
        /*传入了 一个正常的 lambda*/
        fun Base.g() = f()

        /*传入了一个 extension lambda*/
        fun Base.h(x1: Base.() -> Int) = x1()


        val b: Base = Derived()
        b.g() eq 99
        b.h { f() } eq 99
    }

    @Test
    fun `test anonymous function`() {
        fun exec(
            arg1: Int,
            arg2: Int,
            f: Int.(Int) -> Boolean
        ): Boolean = arg1.f(arg2)


        exec(
            10,
            2,
            fun Int.(d: Int): Boolean {
                return this % d == 0
            }
        ) eq true

    }


    @Test
    fun `test StringBuilder`() {
        fun messy(): String {
            val built = StringBuilder()

            built.append("ABCs: ")
            ('a'..'x').forEach { built.append(it) }
            return built.toString()
        }


        fun clean() = buildString {
            append("ABCs: ")
            ('a'..'x').forEach { append(it) }
        }

        fun cleaner() = ('a'..'x').joinToString("", "ABCs: ")

        messy() eq "ABCs: abcdefghijklmnopqrstuvwx"
        messy() eq clean()
        clean() eq cleaner()

    }


    data class User(
        var name: String = "",
        var age: Int = 10,

        ) {
        companion object {
            fun buildUser(
                fillings: User.() -> Unit
            ): User {
                /*you can do anything before this */
                val user = User()

                user.fillings()

                return user
            }
        }
    }

    @Test
    fun `test build`() {

        val u = User.buildUser {
            this.age = 200
            this.name = "carl"
        }

        u eq "User(name=carl, age=200)"
    }
}

