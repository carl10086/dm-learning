package com.duitang.dm.learning.ddd.gw.domain.assemble;

import com.duitang.dm.learning.ddd.gw.domain.core.content.atlas.AtlasCore;
import com.duitang.dm.learning.ddd.gw.domain.core.uact.BizObjectCount;
import com.duitang.dm.learning.ddd.gw.domain.core.uact.UserAct;
import com.duitang.dm.learning.ddd.gw.domain.core.user.UserCore;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class AtlasTinyAssemble {

  /**
   * atlas 的基本信息, 来自于内容系统
   */
  private AtlasCore atlasCore;

  /**
   * atlas 的点赞信息, 来自于 计数系统或者 atlas 系统本身 ..
   */
  private BizObjectCount bizObjectCount;

  /**
   * atlas 作者的基本信息, 来自于 用户系统
   */
  private UserCore author;

  /**
   * 当前登录用户 和 实体的关系, 来自于 用户行为系统
   */
  private UserAct action;

  public AtlasTinyAssemble(AtlasCore atlasCore) {
    this.atlasCore = atlasCore;
  }

  public Long userId() {
    return this.author.id();
  }


  public Long id() {
    return this.atlasCore.id();
  }

  public AtlasTinyAssemble setAtlasCore(AtlasCore atlasCore) {
    this.atlasCore = atlasCore;
    return this;
  }

  public AtlasTinyAssemble setBizObjectCount(BizObjectCount bizObjectCount) {
    this.bizObjectCount = bizObjectCount;
    return this;
  }

  public AtlasTinyAssemble setAuthor(UserCore author) {
    this.author = author;
    return this;
  }

  public AtlasTinyAssemble setAction(UserAct action) {
    this.action = action;
    return this;
  }
}
