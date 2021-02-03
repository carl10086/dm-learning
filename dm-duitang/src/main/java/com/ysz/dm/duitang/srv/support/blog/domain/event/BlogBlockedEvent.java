package com.ysz.dm.duitang.srv.support.blog.domain.event;

import java.util.Date;

public class BlogBlockedEvent extends BlogBaseEvent {

  public BlogBlockedEvent(final Long blogId, final Date occurAt) {
    super(blogId, occurAt);
  }
}
