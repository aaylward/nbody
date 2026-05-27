function percentileSort(arr, p) {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const index = Math.floor(sorted.length * p);
    return sorted[index];
}

class QuickSelect {
    static select(arr, k) {
        let left = 0, right = arr.length - 1;
        while (left < right) {
            let pivotIndex = this.partitionLomuto(arr, left, right);
            if (pivotIndex === k) return arr[k];
            if (k < pivotIndex) right = pivotIndex - 1;
            else left = pivotIndex + 1;
        }
        return arr[k];
    }

    static partitionLomuto(arr, left, right) {
        // Randomize pivot to avoid worst-case O(N^2) on sorted arrays
        const pivotIndex = Math.floor(Math.random() * (right - left + 1)) + left;
        // swap pivot to the end
        let temp = arr[pivotIndex];
        arr[pivotIndex] = arr[right];
        arr[right] = temp;

        const pivot = arr[right];
        let i = left;
        for (let j = left; j < right; j++) {
            if (arr[j] <= pivot) {
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
                i++;
            }
        }
        temp = arr[i];
        arr[i] = arr[right];
        arr[right] = temp;
        return i;
    }
}

function percentileQuickSelect(arr, p) {
    if (arr.length === 0) return 0;
    const copy = [...arr];
    const index = Math.floor(copy.length * p);
    return QuickSelect.select(copy, index);
}

const arr = Array.from({length: 60}, () => Math.random() * 10);

const p1 = percentileSort(arr, 0.95);
const p2 = percentileQuickSelect(arr, 0.95);

console.log(p1, p2, p1 === p2);

let s1 = performance.now();
for(let i=0; i<100000; i++) percentileSort(arr, 0.95);
console.log("sort array", performance.now() - s1);

let s2 = performance.now();
for(let i=0; i<100000; i++) percentileQuickSelect(arr, 0.95);
console.log("quick select", performance.now() - s2);
