package com.ysz.arm.blog.server;

import static org.junit.Assert.*;

import com.linecorp.armeria.client.grpc.GrpcClients;
import com.ysz.arm.blog.client.BlogServiceGrpc.BlogServiceBlockingStub;
import com.ysz.arm.blog.client.CreateBlogPostRequest;
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
        "gjson+http://127.0.0.1:8888/",
        BlogServiceBlockingStub.class
    );
  }

  @Test
  public void createBlog() {
    this.blogService.createBlogPost(CreateBlogPostRequest
                                        .newBuilder()
                                        .setTitle("My first blog")
                                        .setContent("Hello Armeria!")
                                        .build());
  }
}