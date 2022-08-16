package com.ysz.dm.arm.qs;

import com.linecorp.armeria.server.Server;
import com.linecorp.armeria.server.ServerBuilder;
import com.linecorp.armeria.server.docs.DocService;
import com.linecorp.armeria.server.grpc.GrpcService;
import com.ysz.dm.arm.qs.hello.impl.MyHelloService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {

  private static final Logger logger = LoggerFactory.getLogger(Main.class);

  public static void main(String[] args) throws Exception {
    final Server server = newServer(8080);

    server.closeOnJvmShutdown();

    server.start().join();

    logger.info("Server has been started. Serving DocService at http://127.0.0.1:{}/docs", server.activeLocalPort());

  }


  private static Server newServer(int port) {

    final ServerBuilder builder = Server.builder();
    final DocService docService = DocService.builder().exampleRequests(BlogService.class,
                                                                       "createBlogPost",
                                                                       "{\"title\":\"My first blog\", \"content\":\"Hello Armeria!\"}"
    ).build();

    GrpcService grpcService = GrpcService.builder().addService(new MyHelloService()).build();

//    return builder.http(port).annotatedService(new BlogService()).serviceUnder("/docs", docService).build();
    return builder.http(port).service(grpcService).build();
  }
}
