package com.ysz.dm.lib.lang.functions

import org.junit.jupiter.api.Test
import java.util.concurrent.locks.Lock
import java.util.concurrent.locks.ReentrantLock

/**
 * @author carl
 * @create 2022-11-18 2:12 PM
 **/
internal class InLIneFunctionsTest {

    private inline fun <T> inlineSync(lock: Lock, action: () -> T): T {
        lock.lock()
        try {
            return action()
        } finally {
            lock.unlock()
        }
    }


    fun <T> sync(lock: Lock, action: () -> T): T {
        lock.lock()
        try {
            return action()
        } finally {
            lock.unlock()
        }
    }


    @Test
    internal fun `test_sync`() {
        val l = ReentrantLock()
        inlineSync(ReentrantLock()) {
            println(1)
        }
    }
}