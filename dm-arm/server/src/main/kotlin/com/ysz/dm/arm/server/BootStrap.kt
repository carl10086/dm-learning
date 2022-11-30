package com.ysz.dm.arm.server

import com.linecorp.armeria.server.Server
import com.linecorp.armeria.server.grpc.GrpcService

/**
 * @author carl
 * @create 2022-11-28 4:37 PM
 **/
class BootStrap

fun main(args: Array<String>) {
    val sb = Server.builder()

    sb
        .http(8080)
        .blockingTaskExecutor(100)
        .service(
            GrpcService.builder().addService(HelloFacade()).build()
        )


    val server = sb.build()



    server.start().join();

}