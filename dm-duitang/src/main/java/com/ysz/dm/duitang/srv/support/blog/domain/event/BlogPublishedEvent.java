package com.ysz.dm.duitang.srv.support.blog.domain.event;

import java.util.Date;
import lombok.Data;

@Data
public class BlogPublishedEvent {

  private final Date occurAt;

  private final Long blogId;

}
