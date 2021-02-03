package com.ysz.dm.duitang.srv.support.blog.domain.model;

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
  private final BlogUserId blogUserId;

  private final Long fileId; /*为什么不叫 photoId, 代表其实是 oss 的一个文件*/

  /**
   * 保留审核状态、防止 status 变化、导致审核状态丢失
   */
  private BlogMetaAuditStatus auditStatus;

  /**
   * 一个冗余字段 . 就是 Blog 中的 status
   *
   * 个人感觉可以没有 ...
   */
  private BlogStatus status;


  public BlogMeta(
      final Long id,
      final BlogText text,
      final BlogExtra extra,
      final Date createAt,
      final BlogContentType contentType,
      final BlogUserId blogUserId, final Long fileId) {
    this.id = id;
    this.text = text;
    this.extra = extra;
    this.createAt = createAt;
    this.contentType = contentType;
    this.blogUserId = blogUserId;
    this.fileId = fileId;
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
