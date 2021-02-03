package com.ysz.dm.duitang.srv.support.blog.domain.model;

import com.ysz.dm.duitang.srv.infra.anno.DddRootAgg;
import java.util.Date;
import java.util.Objects;
import lombok.Getter;

@DddRootAgg
public class Blog {

  @Getter
  private final Long id;

  @Getter
  private final BlogMeta blogMeta;

  private BlogStatus finalStatus;

  private final Long parentId;

  private final BlogUserId blogUserId;

  private final Date createAt;

  public Blog(
      final Long id,
      final BlogMeta blogMeta,
      final Long parentId,
      final BlogUserId blogUserId,
      final Date createAt) {
    this.id = id;
    this.blogMeta = blogMeta;
    this.createAt = createAt;
    this.finalStatus = BlogStatus.waiting_audit;
    this.parentId = parentId;
    this.blogUserId = blogUserId;
  }

  public Long rootId() {
    return this.blogMeta.getId();
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

  /**
   *
   * @param curMsg 可以理解为审核人员眼中的 msg
   * @param curStatus 可以理解
   * @param now 当前时间
   * @return
   */
  public boolean block(final String curMsg, final Integer curStatus, final Date now) {
    // 1. chkMetaMsg is curMsg  !
    // 2. chkBlogStatus is curStatus !
    // 3. chkBlogStatus can to blogStatus !

    // 4. doBlock();
    return false;
  }
}
