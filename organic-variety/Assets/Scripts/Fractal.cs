using UnityEngine;
using Unity.Burst;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Jobs;

using static Unity.Mathematics.math;
using quaternion = Unity.Mathematics.quaternion;
using Random = UnityEngine.Random;


public class Fractal : MonoBehaviour
{
    [SerializeField, Range(3, 10)] int depth = 4;
    [SerializeField] Mesh mesh, leafMesh;
    [SerializeField] Material material;
    [SerializeField] Gradient gradientA, gradientB;
    [SerializeField] Color leafColorA, leafColorB;
    [SerializeField, Range(0,90f)] float maxSagAngleA = 15f, maxSagAngleB = 25f;
    [SerializeField, Range(0,90f)] float spinSpeedA = 20f, spinSpeedB = 25f;
    
    [SerializeField, Range(0f,1f)] float reverseSpinChance = 0.25f;
    
    
    static quaternion[] rotations = { 
        quaternion.identity,
        quaternion.RotateZ(-0.5f * PI), quaternion.RotateZ(0.5f * PI),
        quaternion.RotateX(0.5f * PI), quaternion.RotateX(-0.5f * PI)
    };
    
    struct FractalPart
    {
        public float3 worldPosition;
        public quaternion rotation, worldRotation;
        public float spinAngle, maxSagAngle, spinVelocity;
    }
    
    NativeArray<FractalPart>[] parts; //parts seperated by layer
    NativeArray<float3x4>[] matrices;

    ComputeBuffer[] matricesBuffer;
    static MaterialPropertyBlock propertyBlock;

    static readonly int matriciesId = Shader.PropertyToID("_Matrices"),
        sequenceNumbersID = Shader.PropertyToID("_SequenceNumbers"),
        colorAId = Shader.PropertyToID("_ColorA"),
        colorBId = Shader.PropertyToID("_ColorB");

    Vector4[] sequenceNumbers;
    
    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast, CompileSynchronously = true)]
    struct UpdateFractalLevelJob : IJobFor
    {
        public float scale;
        public float deltaTime;

        public NativeArray<FractalPart> parts;
        
        [ReadOnly]
        public NativeArray<FractalPart> parents;
        [WriteOnly]
        public NativeArray<float3x4> matrices;
        public void Execute(int i) {
            FractalPart parent = parents[i / 5];
            FractalPart part = parts[i];
            part.spinAngle += deltaTime * part.spinVelocity;
                
            float3 upAxis = mul(mul(parent.worldRotation, part.rotation), up());
            float3 sagAxis = cross(up(), upAxis);
            float sagMagnitude = length(sagAxis);
            quaternion baseRotation;
            if (sagMagnitude > 0f) {
                sagAxis /= sagMagnitude; //normalize vector
                quaternion sagRotation = quaternion.AxisAngle(sagAxis, part.maxSagAngle * sagMagnitude);
                baseRotation = mul(sagRotation, parent.worldRotation);
            } else {
                baseRotation = parent.worldRotation;
            }
            
            part.worldRotation = mul(baseRotation, mul(part.rotation, quaternion.RotateY(part.spinAngle)));
            part.worldPosition = 
                parent.worldPosition + 
                mul(part.worldRotation,
                (float3(0, 1.5f * scale, 0)));
            parts[i] = part;
            float3x3 r = float3x3(part.worldRotation) * scale;
            matrices[i] = float3x4(r.c0, r.c1, r.c2, part.worldPosition);
        }
    }
    
    FractalPart CreatePart(int childIndex) => new FractalPart {
        rotation = rotations[childIndex],
        spinVelocity = 
            (Random.value < reverseSpinChance ? -1f : 1f) * 
            radians(Random.Range(spinSpeedA, spinSpeedB)),
        maxSagAngle = radians(Random.Range(maxSagAngleA, maxSagAngleB))
    };
    
    void OnEnable() {
        parts = new NativeArray<FractalPart>[depth];
        matrices = new NativeArray<float3x4>[depth];
        matricesBuffer = new ComputeBuffer[depth];
        sequenceNumbers = new Vector4[depth];
        int stride = 12 * 4;
        for (int i = 0, length = 1; i < parts.Length; i++, length *= 5) {
            parts[i] = new NativeArray<FractalPart>(length, Allocator.Persistent);
            matrices[i] = new NativeArray<float3x4>(length, Allocator.Persistent);
            matricesBuffer[i] = new ComputeBuffer(length, stride);
            sequenceNumbers[i] = new Vector4(Random.value, Random.value, Random.value, Random.value);
        }

        parts[0][0] = CreatePart(0);
        for (int li = 1; li < parts.Length; li++) {
            NativeArray<FractalPart> levelPart = parts[li];
            for (int fpi = 0; fpi < levelPart.Length; fpi+=5) {
                for (int ci = 0; ci < 5; ci++) {
                    levelPart[fpi + ci] = CreatePart(ci);
                }
            }
        }

        propertyBlock ??= new MaterialPropertyBlock();
    }

    void OnDisable() {
        for (int i = 0; i < matricesBuffer.Length; i++) {
            matricesBuffer[i].Release();
            parts[i].Dispose();
            matrices[i].Dispose();
        }
        matrices = null;
        parts = null;
        matricesBuffer = null;
        sequenceNumbers = null;
    }

    
    void OnValidate() {
        if (parts != null && enabled) {
            OnEnable();
            OnDisable();
        }
    }

    void Update() {
        float deltaTime = Time.deltaTime;
        FractalPart rootPart = parts[0][0];
        rootPart.spinAngle += rootPart.spinVelocity * deltaTime;
        rootPart.worldRotation = mul( transform.rotation , mul(rootPart.rotation , quaternion.RotateY(rootPart.spinAngle)));
        rootPart.worldPosition = transform.position;
        parts[0][0] = rootPart;
        float objectScale = transform.lossyScale.x;
        float scale = objectScale;
        float3x3 r = float3x3(rootPart.worldRotation) * scale;
        matrices[0][0] = float3x4(r.c0, r.c1, r.c2, rootPart.worldPosition);
        
        JobHandle handle = default;
        for (int li = 1; li < parts.Length; li++) {
            scale *= 0.5f;
            handle = new UpdateFractalLevelJob
            {
                deltaTime = deltaTime,
                scale = scale,
                parts = parts[li],
                parents = parts[li-1],
                matrices = matrices[li]
            }.ScheduleParallel(parts[li].Length,  5,handle);
        }
        handle.Complete();
        
        int leafIndex = matricesBuffer.Length - 1;
        Mesh instanceMesh;
        Bounds bounds = new Bounds(Vector3.zero, 3f * Vector3.one);
        for (int i = 0; i < matricesBuffer.Length; i++) {
            ComputeBuffer buffer = matricesBuffer[i];
            buffer.SetData(matrices[i]);
            propertyBlock.SetBuffer(matriciesId, buffer);
            Color colorA, colorB;
            if (i == leafIndex) {
                colorA = leafColorA;
                colorB = leafColorB;
                instanceMesh = leafMesh;
            } else {
                float gradientInterpolator = i / (matricesBuffer.Length - 2f);
                colorA = gradientA.Evaluate(gradientInterpolator);
                colorB = gradientB.Evaluate(gradientInterpolator);
                instanceMesh = mesh;
            }
            propertyBlock.SetColor(colorAId,  colorA);
            propertyBlock.SetColor(colorBId,  colorB);
            propertyBlock.SetVector(sequenceNumbersID, sequenceNumbers[i]);
            Graphics.DrawMeshInstancedProcedural(instanceMesh, 0, material, bounds, buffer.count, propertyBlock);
        }
    }
}