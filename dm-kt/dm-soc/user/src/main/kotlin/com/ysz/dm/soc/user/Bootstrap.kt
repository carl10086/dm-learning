package com.ysz.dm.soc.user

import org.springframework.boot.SpringApplication
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.autoconfigure.cassandra.CassandraAutoConfiguration
import org.springframework.boot.autoconfigure.data.mongo.MongoDataAutoConfiguration
import org.springframework.boot.autoconfigure.data.redis.RedisAutoConfiguration
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration
import org.springframework.boot.autoconfigure.mongo.MongoAutoConfiguration

/**
 * @author carl
 * @since 2023-02-22 12:46 AM
 **/
@SpringBootApplication(
    exclude = [
        RedisAutoConfiguration::class,
        MongoDataAutoConfiguration::class,
        MongoAutoConfiguration::class,
        CassandraAutoConfiguration::class,
        DataSourceAutoConfiguration::class,
    ]
)
open class Bootstrap

fun main(args: Array<String>) {
    SpringApplication.run(Bootstrap::class.java, *args)
}