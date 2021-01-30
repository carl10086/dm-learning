package com.ysz.dm.duitang.srv.support.blog.domain;

import java.util.Date;
import java.util.Objects;
import lombok.Getter;

public class BlogForward {

  @Getter
  private final Long id;

  @Getter
  private final BlogMeta blogMeta;

  private BlogForwardFinalStatus finalStatus;

  private final Long parentId;

  private final BlogUser blogUser;

  private final Date createAt;

  public BlogForward(
      final Long id,
      final BlogMeta blogMeta,
      final Long parentId,
      final BlogUser blogUser,
      final Date createAt) {
    this.id = id;
    this.blogMeta = blogMeta;
    this.createAt = createAt;
    this.finalStatus = BlogForwardFinalStatus.waiting_audit;
    this.parentId = parentId;
    this.blogUser = blogUser;
  }

  /**
   * 判断当前的 转发关系 id 是不是原发
   * @return true：当前的 forward 是原发, false: 当前的 forward 不是原发
   */
  public boolean isOrigin() {
    return Objects.equals(getId(), blogMeta.getId());
  }


  /**
   * 核心业务逻辑、是否可以进行转发
   * @return true: 表示可以转发、 false 表示不可以
   */
  public boolean canBeForward() {
    if (!finalStatus.canBeForward()) {
      return false;
    }

    /*如果是非原发、还要判断原发状态*/
    if (!isOrigin()) {
      return blogMeta.canBeForward();
    }

    return true;
  }
}
