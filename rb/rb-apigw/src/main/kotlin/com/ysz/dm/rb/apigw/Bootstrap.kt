package com.ysz.dm.rb.apigw

import org.springframework.boot.SpringApplication
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration
import org.springframework.context.annotation.EnableAspectJAutoProxy

/**
 * <pre>
 * class desc here
</pre> *
 *
 * @author carl.yu
 * @createAt 2022/11/12
 */
@SpringBootApplication(exclude = [SecurityAutoConfiguration::class])
@EnableAspectJAutoProxy(proxyTargetClass = true)
open class Bootstrap {}

fun main(args: Array<String>) {
    SpringApplication.run(Bootstrap::class.java, *args)
}