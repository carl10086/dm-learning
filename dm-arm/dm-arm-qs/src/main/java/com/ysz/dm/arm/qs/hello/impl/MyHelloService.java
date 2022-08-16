package com.ysz.dm.arm.qs.hello.impl;

import com.ysz.dm.arm.qs.hello.Hello.HelloReply;
import com.ysz.dm.arm.qs.hello.Hello.HelloRequest;
import com.ysz.dm.arm.qs.hello.HelloServiceGrpc.HelloServiceImplBase;
import io.grpc.stub.StreamObserver;

public class MyHelloService extends HelloServiceImplBase {

  @Override
  public void hello(HelloRequest request, StreamObserver<HelloReply> responseObserver) {
//    super.hello(request, responseObserver);
    HelloReply reply = HelloReply.newBuilder()
        .setMessage("Hello, " + request.getName() + '!')
        .build();
    responseObserver.onNext(reply);
    responseObserver.onCompleted();
  }
}
