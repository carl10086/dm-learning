package com.ysz.dm.rb.user.infra.rpc

import com.ysz.dm.rb.user.infra.base.armeria.ArmeriaGrpc
import com.ysz.dm.rb.user.user.HelloReq
import com.ysz.dm.rb.user.user.HelloResp
import com.ysz.dm.rb.user.user.UserFacadeGrpc
import io.grpc.stub.StreamObserver

/**
 * @author carl
 * @create 2022-11-16 5:43 PM
 **/
@ArmeriaGrpc
class BlogFacade : UserFacadeGrpc.UserFacadeImplBase() {
    override fun hello(request: HelloReq, responseObserver: StreamObserver<HelloResp>) {
        responseObserver.onNext(HelloResp.newBuilder().apply { name = request.id.toString() }.build())
        responseObserver.onCompleted()
    }
}