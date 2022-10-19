package com.ysz.arm.blog.server;

import com.google.common.base.Strings;
import com.linecorp.armeria.common.RequestContext;
import com.ysz.arm.blog.client.BlogPost;
import com.ysz.arm.blog.client.BlogServiceGrpc;
import com.ysz.arm.blog.client.CreateBlogPostRequest;
import com.ysz.arm.blog.server.config.ArmeriaGrpc;
import com.ysz.arm.blog.server.core.exceptions.CustomInvalidException;
import io.grpc.stub.StreamObserver;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import lombok.extern.slf4j.Slf4j;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/10/15
 **/
@Slf4j
@ArmeriaGrpc
public class BlogService extends BlogServiceGrpc.BlogServiceImplBase {

  private final AtomicInteger idGenerator = new AtomicInteger();
  private final Map<Integer, BlogPost> blogPosts = new ConcurrentHashMap<>();

  @Override
  public void createBlogPost(CreateBlogPostRequest request, StreamObserver<BlogPost> responseObserver) {
    RequestContext requestContext = RequestContext.currentOrNull();
    if (log.isDebugEnabled()) {
      log.debug("createBlogPost:{}", request);
    }

    /*1. check title*/
    if (Strings.isNullOrEmpty(request.getTitle())) {
//      com.google.rpc.Status status = com.google.rpc.Status.newBuilder()
//          .setCode(Code.INVALID_ARGUMENT.getNumber())
//          .setMessage("title is empty or null")
//          .build();
//      responseObserver.onError(StatusProto.toStatusRuntimeException(status));
      throw new CustomInvalidException("title is empty or null");
    }

    final int id = idGenerator.getAndIncrement();
    final Instant now = Instant.now();
    final BlogPost updated = BlogPost.newBuilder().setId(id).setTitle(request.getTitle())
        .setContent(request.getContent()).setModifiedAt(now.toEpochMilli()).setCreatedAt(now.toEpochMilli()).build();
    blogPosts.put(id, updated);
    final BlogPost stored = updated;
    responseObserver.onNext(stored);
    responseObserver.onCompleted();

  }
}
