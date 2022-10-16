package com.ysz.arm.blog.server;

import com.google.rpc.Code;
import com.google.rpc.Status;
import com.linecorp.armeria.client.grpc.GrpcClients;
import com.linecorp.armeria.common.grpc.StatusCauseException;
import com.ysz.arm.blog.client.BlogPost;
import com.ysz.arm.blog.client.BlogServiceGrpc.BlogServiceBlockingStub;
import com.ysz.arm.blog.client.CreateBlogPostRequest;
import io.grpc.StatusRuntimeException;
import io.grpc.protobuf.StatusProto;
import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.Test;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/10/15
 **/
@Slf4j
public class BlogServiceTest {

  private BlogServiceBlockingStub blogService;

  @Before
  public void setUp() throws Exception {
    this.blogService = GrpcClients.newClient(
        "gjson+http://127.0.0.1:10005/",
        BlogServiceBlockingStub.class
    );
  }

  @Test
  public void createBlog() {
    try {
      BlogPost blogPost = this.blogService.createBlogPost(CreateBlogPostRequest
                                                              .newBuilder()
//                                        .setTitle("My first blog")
                                                              .setContent("Hello Armeria!")
                                                              .build());

      log.info("success called:{}", blogPost);
    } catch (Throwable e) {
      Status status = StatusProto.fromThrowable(e);
      log.error("call failed, code:{}, message:{}", Code.forNumber(status.getCode()), status.getMessage());

      /*this code can show origin verbose exception*/
      if (e instanceof StatusRuntimeException) {
        Throwable cause = e.getCause();
        if (cause instanceof StatusCauseException) {
          StatusCauseException causeException = (StatusCauseException) cause;

          String originalClassName = ((StatusCauseException) cause).getOriginalClassName();
          String originalMessage = causeException.getOriginalMessage();
          log.error("className:{}, message:{}", originalClassName, originalMessage);
        }
      }
    }

  }
}