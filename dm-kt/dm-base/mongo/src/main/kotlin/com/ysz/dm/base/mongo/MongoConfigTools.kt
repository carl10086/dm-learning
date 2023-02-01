package com.ysz.dm.base.mongo

import com.mongodb.MongoClientSettings
import com.mongodb.ReadPreference
import com.mongodb.ServerAddress
import com.mongodb.client.MongoClient
import com.mongodb.client.MongoClients
import com.mongodb.connection.ConnectionPoolSettings
import org.springframework.data.mongodb.MongoDatabaseFactory
import org.springframework.data.mongodb.core.MongoTemplate
import org.springframework.data.mongodb.core.SimpleMongoClientDatabaseFactory
import org.springframework.data.mongodb.core.convert.*
import org.springframework.data.mongodb.core.mapping.MongoMappingContext

/**
 * @author carl
 * @since 2023-02-01 2:08 PM
 **/
object MongoConfigTools {

    fun buildMongoTpl(client: MongoClient, database: String): MongoTemplate =
        SimpleMongoClientDatabaseFactory(client, database).let {
            MongoTemplate(it, buildMappingMongoConvertByDefault(it))
        }


    fun buildClient(urls: List<String>): MongoClient =
        buildClient(servers = urls.map(::urlToServerHost)) { }


    private fun urlToServerHost(url: String): ServerAddress {
        val hostAndPort = url.split(":")
        return ServerAddress(hostAndPort[0], hostAndPort[1].toInt())
    }


    private fun buildMappingMongoConvertByDefault(
        factory: MongoDatabaseFactory
    ): MappingMongoConverter {
        val dbRefResolver: DbRefResolver = DefaultDbRefResolver(factory)
        val conversions = MongoCustomConversions(emptyList<Any>())
        val mappingContext = MongoMappingContext()
        mappingContext.setSimpleTypeHolder(conversions.simpleTypeHolder)
        mappingContext.afterPropertiesSet()
        val converter = MappingMongoConverter(dbRefResolver, mappingContext)
        converter.customConversions = conversions
        converter.setCodecRegistryProvider(factory)
        // type key 设置为 null 可以禁用 _class 字段
        converter.typeMapper = DefaultMongoTypeMapper(null, converter.mappingContext)
        converter.afterPropertiesSet()
        return converter
    }


    fun buildClient(
        servers: List<ServerAddress>,
        filling: MongoClientSettings.Builder.() -> Unit
    ): MongoClient {
        val builder = MongoClientSettings.builder()
        builder.applyToClusterSettings {
            it.hosts(servers)
        }

        builder.applyToConnectionPoolSettings { it.applySettings(ConnectionPoolSettings.builder().maxSize(16).build()) }
        builder.readPreference(ReadPreference.secondaryPreferred())

        /*1. we set default values here*/

        builder.filling()

        return MongoClients.create(builder.build())
    }


}