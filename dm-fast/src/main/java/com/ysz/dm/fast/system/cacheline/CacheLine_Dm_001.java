package com.ysz.dm.fast.system.cacheline;

/**
 * 例子:
 *
 *  32 bit 系统;
 *  L1 Data Cache: 32 Bytes
 *  Cache Line: 64 Bytes
 *  8 Way means: One Cache Line Group Contains 8 Cache Line .
 *
 *  SO:
 *
 *  L1 Data Cache: 64 Groups ;
 *
 *
 */
public class CacheLine_Dm_001 {

  /**
   * mapping For main Memory and  Cache memory .
   */
  private static class CacheLineVmAddress {

    /**
     * 32 bit 的 int 数字;
     *
     * 前 24 bit 拷贝自内存地址的前 24 bit
     * 中间 6 it 表示 group 的 索引 .
     * 最后 6 bit 表示 line 中的 索引 .
     */
    private int vmOffset;

    public int cacheLineGroupOffset() {
      // todo 取一个 int 的中间 6 bits、也就是 24,25,26,27,28,29
      return 0;
    }

    public int cacheLineOffset() {
      // todo 取一个 int 的 后 6 bits
      return 0;
    }

    /**
     * 24 bit 的 tag
     */
    public int tag() {
      // todo 取一个 int 的前 24 bits
      return 0;
    }
  }

  /**
   * 数据包装类、代表一个 Byte
   */
  private static class DataWrapper {

    private Byte data;
    private CacheLineVmAddress cacheLineVmAddress;

    public int tag() {
      return this.cacheLineVmAddress.tag();
    }

  }

  private static class CacheLine {

    // 64 Byte
    private DataWrapper[] datas = new DataWrapper[64];

    public int firstTag() {
      return datas[0].tag();
    }

    public DataWrapper getByLineOffset(final int lineOffset) {
      return datas[lineOffset];
    }
  }


  private static class CacheLineGroup {

    private CacheLine[] lines = new CacheLine[8]; // 每组有 8个 line

    public DataWrapper searchData(CacheLineVmAddress cacheLineVmAddress) {
      int targetTag = cacheLineVmAddress.tag();
      for (int i = 0; i < lines.length; i++) /*进行 O(n) 的循环*/ {
        if (targetTag == lines[i].firstTag()) {
          /*cache hit*/
          return lines[i].getByLineOffset(cacheLineVmAddress.cacheLineOffset());
        }
      }

      /*cache miss*/
      return null;
    }
  }

  /**
   * L1 缓存对象
   */
  private static class LevelOneCache {


    private CacheLineGroup[] groups = new CacheLineGroup[64]; // 2 的 6次方 . . . .

    private DataWrapper find(CacheLineVmAddress cacheLineVmAddress) {
      CacheLineGroup group = quickFindGroup(cacheLineVmAddress);
      return group.searchData(cacheLineVmAddress);
    }

    // 要方便查询、只需要构造一个对象 CacheLineVmAddress .
    private CacheLineGroup quickFindGroup(final CacheLineVmAddress cacheLineVmAddress) {
      final int idx = cacheLineVmAddress.cacheLineGroupOffset();
      CacheLineGroup group = groups[idx];
      return group;
    }
  }

}
