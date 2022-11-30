package com.ysz.dm.arm.client

import com.google.protobuf.StringValue
import com.linecorp.armeria.client.ClientFactory
import com.linecorp.armeria.client.grpc.GrpcClients
import com.ysz.dm.arm.client.decorators.CustomHttpDecorator
import com.ysz.dm.arm.hello.HelloFacadeGrpc
import org.slf4j.LoggerFactory

/**
 * @author carl
 * @create 2022-11-28 4:50 PM
 **/
internal class HelloFacadeClientTest {


    @org.junit.jupiter.api.Test
    internal fun `test_hello`() {

        val factory: ClientFactory = ClientFactory
            .builder()
            .build()

        val helloFacadeClient = GrpcClients.builder("gproto+http://127.0.0.1:8080")
            .factory(factory)
            .decorator(CustomHttpDecorator.newDecorator())
            .build(HelloFacadeGrpc.HelloFacadeBlockingStub::class.java)
        log.info("start hello")
        val hello = helloFacadeClient.hello(StringValue.of("carl"))
        log.info("finish hello:{}", hello)
    }


    companion object {
        private val log = LoggerFactory.getLogger(HelloFacadeClientTest::class.java)
    }

}
