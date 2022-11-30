package com.ysz.dm.lib.lang.juc.custom

import org.slf4j.LoggerFactory
import java.time.LocalDate
import java.util.concurrent.Future
import java.util.concurrent.ScheduledThreadPoolExecutor

/**
 * @author carl
 * @create 2022-11-28 5:30 PM
 **/
class CustomScheduledThreadPoolExecutor(numThreads: Int) : ScheduledThreadPoolExecutor(numThreads) {

    override fun allowCoreThreadTimeOut(value: Boolean) {
    }

    override fun submit(task: Runnable): Future<*> {
        /*this is called by parent thread*/
        logger.info("now:{}", LocalDate.now())
        return super.submit(task)
    }

    companion object {
        private val logger = LoggerFactory.getLogger(CustomScheduledThreadPoolExecutor::class.java)
    }
}