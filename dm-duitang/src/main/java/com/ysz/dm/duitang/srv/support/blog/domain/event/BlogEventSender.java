package com.ysz.dm.duitang.srv.support.blog.domain.event;

import com.ysz.dm.duitang.srv.support.blog.domain.model.Blog;

public interface BlogEventSender {

  void sendBlogPublishedEvent(Blog blog);

  void sendBlogBlockedEvent(Blog blog);
}
