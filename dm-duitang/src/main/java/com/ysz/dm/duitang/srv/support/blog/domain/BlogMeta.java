package com.ysz.dm.duitang.srv.support.blog.domain;

import java.util.Date;
import lombok.Getter;
import lombok.Setter;

public class BlogMeta {

  @Getter
  private final Long id;

  private BlogText text;

  /**
   *
   * 已经废弃没有用的东西、但是为了兼容老版本 保留写入, 不建议用来查询
   */
  @Setter
  private final BlogExtra extra;

  @Getter
  private final Date createAt;

  private final BlogContentType contentType;

  @Getter
  private final BlogUser blogUser;

  private final Long photoId;

  private BlogMetaAuditStatus auditStatus;


  public BlogMeta(
      final Long id,
      final BlogText text,
      final BlogExtra extra,
      final Date createAt,
      final BlogContentType contentType,
      final BlogUser blogUser, final Long photoId) {
    this.id = id;
    this.text = text;
    this.extra = extra;
    this.createAt = createAt;
    this.contentType = contentType;
    this.blogUser = blogUser;
    this.photoId = photoId;
    this.auditStatus = BlogMetaAuditStatus.waiting_audit;
  }

  public BlogMeta setText(final BlogText text) {
    this.text = text;
    return this;
  }

  @Deprecated
  public BlogExtra getExtra() {
    return extra;
  }

  public boolean canBeForward() {
    return auditStatus.canBeForward();
  }
}
