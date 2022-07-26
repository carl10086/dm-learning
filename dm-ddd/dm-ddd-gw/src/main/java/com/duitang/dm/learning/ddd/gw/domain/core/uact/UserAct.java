package com.duitang.dm.learning.ddd.gw.domain.core.uact;

import com.duitang.dm.learning.ddd.gw.domain.constant.BizObjectType;
import lombok.Getter;
import lombok.ToString;

/**
 * 用户和实体的交互关系.
 *
 * 例如 用户是否点赞过 atlas, 用户是否收藏过 atlas , 用户是否评论过 atlas 都封装在这里
 */
@ToString
@Getter
public class UserAct {

  private Long userId;
  private Long objectId;

  private BizObjectType bizObjectType;
  private Boolean userLike;
  private Boolean userComment;
  private Boolean userFav;


  public UserAct setUserId(Long userId) {
    this.userId = userId;
    return this;
  }

  public UserAct setObjectId(Long objectId) {
    this.objectId = objectId;
    return this;
  }

  public UserAct setBizObjectType(BizObjectType bizObjectType) {
    this.bizObjectType = bizObjectType;
    return this;
  }

  public UserAct setUserLike(Boolean userLike) {
    this.userLike = userLike;
    return this;
  }

  public UserAct setUserComment(Boolean userComment) {
    this.userComment = userComment;
    return this;
  }

  public UserAct setUserFav(Boolean userFav) {
    this.userFav = userFav;
    return this;
  }
}
