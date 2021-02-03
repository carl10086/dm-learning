package com.ysz.dm.duitang.srv.support.blog.domain.event;

import java.util.Date;
import lombok.Data;

@Data
public class BlogBaseEvent {

  private final Long blogId;
  private final Date occurAt;


}
