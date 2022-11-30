package com.ysz.dm.arm.server

import com.google.protobuf.StringValue
import com.ysz.dm.arm.hello.HelloFacadeGrpc
import io.grpc.stub.StreamObserver

/**
 * @author carl
 * @create 2022-11-28 4:41 PM
 **/
open class HelloFacade : HelloFacadeGrpc.HelloFacadeImplBase() {

    override fun hello(request: StringValue, responseObserver: StreamObserver<StringValue>) {
        responseObserver.onNext(StringValue.of("hello:${request.value}"))
        responseObserver.onCompleted()
    }
}