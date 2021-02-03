package com.ysz.dm.duitang.srv.support.blog.port.adapter.event;

import com.ysz.dm.duitang.cli.support.blog.dto.BlogEventDTO;
import com.ysz.dm.duitang.cli.support.blog.dto.BlogEventType;
import com.ysz.dm.duitang.srv.support.blog.domain.event.BlogEventSender;
import com.ysz.dm.duitang.srv.support.blog.domain.model.Blog;

public class BlogEventSenderImpl implements BlogEventSender {


  @Override
  public void sendBlogPublishedEvent(final Blog blog) {
    BlogEventDTO blogEventDTO = new BlogEventDTO();
    blogEventDTO.setEventType(BlogEventType.publish.name());
    blogEventDTO.setOccurAt(blog.getBlogMeta().getCreateAt());
    blogEventDTO.setBlogId(blog.getId());
    sendKafkaEvent(blogEventDTO);
  }

  @Override
  public void sendBlogBlockedEvent(final Blog blog) {

  }

  private void sendKafkaEvent(
      final BlogEventDTO blogEventDTO) {
  }
}
