package com.ysz.dm.ysz.base.mysql.config.dynamic

import org.aspectj.lang.annotation.Around
import org.aspectj.lang.annotation.Aspect
import org.aspectj.lang.annotation.Pointcut
import org.slf4j.LoggerFactory

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2023/1/31
 **/
@Aspect
open class DynamicDataSourceAspect {
    @Pointcut("execution(* (@com.ysz.dm.base.core.spring.ddd.CommandHandler *).*(..))")
    open fun command() {
    }

//    @Pointcut("execution(* (@com.ysz.dm.base.core.spring.ddd.QueryHandler *).*(..))")
//    open fun query() {
//    }

    @Around("command()")
    open fun aroundCommand() {
        if (log.isDebugEnabled) log.debug("aroundCommand")

        return try {
            DynamicDataSourceHolder.forcePrimary()
        } finally {
            DynamicDataSourceHolder.reset()
        }
    }


    companion object {
        private val log = LoggerFactory.getLogger(DynamicDataSourceAspect::class.java)
    }

}