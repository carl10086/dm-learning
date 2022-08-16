package com.ysz.dm.arm.qs;

import com.linecorp.armeria.common.HttpRequest;
import com.linecorp.armeria.common.HttpResponse;
import com.linecorp.armeria.server.annotation.Post;
import com.linecorp.armeria.server.annotation.RequestConverter;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class BlogService {

  private final Map<Integer, BlogPost> blogPosts = new ConcurrentHashMap<>();

  /**
   * Creates a {@link BlogPost} from an {@link HttpRequest}. The {@link HttpRequest} is converted into {@link BlogPost}
   * by the {@link BlogPostRequestConverter}.
   */
  @Post("/blogs")
  @RequestConverter(BlogPostRequestConverter.class)
  public HttpResponse createBlogPost(BlogPost blogPost) {
    // Use a map to store the blog. In real world, you should use a database.
    blogPosts.put(blogPost.getId(), blogPost);

    // Send the created blog post as the response.
    // We can add additional property such as a url of
    // the created blog post.(e.g. "http://tutorial.com/blogs/0") to respect the Rest API.
    return HttpResponse.ofJson(blogPost);
  }

}
