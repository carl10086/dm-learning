package com.ysz.dm.ddd.vshop.domain.core.inventory;

import com.ysz.dm.ddd.vshop.domain.core.common.media.PictureId;
import java.util.List;
import lombok.Getter;
import lombok.ToString;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/9/9
 **/
@ToString
@Getter
public class InventoryPictures {

  private PictureId mainPictureId;

  private List<PictureId> pictureIds;

}
