package com.ysz.dm.duitang.srv.support.blog.domain.adapter;

import com.ysz.dm.duitang.srv.infra.anno.DddEntity;
import lombok.Setter;

@DddEntity
public class BlogUser {

  @Setter
  private Long userId;
  @Setter
  private Integer age;
  @Setter
  private Long photoId;
  @Setter
  private String username;

}
