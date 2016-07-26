package DataStructure;

public class indexTrie {

	public long offset;
	public indexTrie[] children;

	public indexTrie() {
		offset = -1;
		children = null;
	}	

	public void set(long val) {
		offset = val;
	}

	public long get() {
		return offset;
	}

}
