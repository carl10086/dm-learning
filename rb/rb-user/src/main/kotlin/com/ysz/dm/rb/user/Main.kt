package com.ysz.dm.rb.user

import org.springframework.boot.SpringApplication
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/15
 **/

@SpringBootApplication(
    exclude = [DataSourceAutoConfiguration::class]
//        RedisAutoConfiguration::class.java,
//        MongoDataAutoConfiguration::class.java,
//        MongoAutoConfiguration::class.java,
//        MongoRepositoriesAutoConfiguration::class.java,
//        DataSourceAutoConfiguration::class.java,
//        JacksonAutoConfiguration::class.java,
//        RedisAutoConfiguration::class.java,
//        RedisRepositoriesAutoConfiguration::class.java,
//        SolrAutoConfiguration::class.java,
//        CassandraAutoConfiguration::class.java
)
open class Main {
}

fun main(args: Array<String>) {
    SpringApplication.run(Main::class.java, *args)
}

