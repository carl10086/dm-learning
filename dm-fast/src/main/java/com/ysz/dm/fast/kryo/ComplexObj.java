package com.ysz.dm.fast.kryo;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import lombok.Data;

/**
 * @author carl.yu
 * @date 2020/3/17
 */
@Data
public class ComplexObj implements Serializable {

  private String id;

  private String rootId;

  private String parentId;

  /**
   * 开始时间、秒
   */
  private int startAt;

  /**
   * 持续时间、秒
   */
  private int duration;

  /**
   * app 名称
   */
  private String app;

  /**
   * 事件名称
   */
  private String name;
  /**
   * 逻辑分组
   */
  private String type;

  private List<ComplexObj> children;

  public void addChild(ComplexObj child) {
    children.add(child);
  }


  public List<ComplexObj> asPlainObjs() {
    List<ComplexObj> result = new ArrayList<>();
    addToResult(this, result, this.getRootId());
    return result;
  }

  private void addToResult(ComplexObj obj, List<ComplexObj> result, String rootId) {
    obj.setRootId(rootId);
    result.add(obj);

    if (obj.getChildren() != null) {
      for (ComplexObj child : obj.getChildren()) {
        child.setParentId(obj.getId());
        addToResult(child, result, rootId);
      }
    }
  }
}
