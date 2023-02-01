package com.ysz.dm.base.mongo

import com.mongodb.ReadPreference
import com.mongodb.ServerAddress
import com.mongodb.client.internal.MongoClientImpl
import com.ysz.dm.base.mongo.MongoConfigTools.buildClient
import com.ysz.dm.base.test.eq
import org.junit.jupiter.api.Test

/**
 * <pre>
 * class desc here
</pre> *
 * @author carl.yu
 * @since 2023/2/2
 */
internal class MongoConfigToolsTest {
    @Test
    fun `test build`() {
        val settings = (buildClient(listOf("127.0.0.1:27017")) as MongoClientImpl).settings
        settings.clusterSettings.hosts eq listOf(ServerAddress("127.0.0.1", 27017))
        settings.readPreference eq ReadPreference.secondaryPreferred()
        settings.connectionPoolSettings.maxSize eq 16
    }

    data class A(val name: String, val male: Boolean = true)

    @Test
    fun `test eq`() {
        /*1. test object eq*/
        A("aa") eq A("aa")

        /*2. test primitive eq*/
        val a = 1
        a eq 1

        /*3. test primitive array eq*/
        intArrayOf(1, 2, 3) eq intArrayOf(1, 2, 3)

        /*4. test primitive array eq*/
        arrayOf(A("a")) eq arrayOf(A("a"))

        /*5. test list eq*/
        listOf(A("aa")) eq listOf(A("aa"))
    }
}