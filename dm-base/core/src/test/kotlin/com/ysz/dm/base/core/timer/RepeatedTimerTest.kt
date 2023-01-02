package com.ysz.dm.base.core.timer

import org.slf4j.LoggerFactory

/**
 * <pre>
 * class desc here
</pre> *
 * @author carl.yu
 * @since 2023/1/3
 */
internal class RepeatedTimerDm


fun main() {
    val timer = HashedWheelTimer()

    val log = LoggerFactory.getLogger(RepeatedTimerDm::class.java)
    val repeat = object : RepeatedTimer("test", 1000, timer) {
        override fun onTrigger() {
            log.info("trigger")
        }
    }

    repeat.start()

    Thread {
        Thread.sleep(5000L)
        repeat.stop()
    }.start()
}