也就是说，我们可以先预测s，然后传入s来预测该s对应的o，然后传入s、o来预测所传入的s、o的关系p，实际应用中，我们还可以把o、p的预测合并为一步，所以总的步骤只需要两步：==先预测s，然后传入s来预测该s所对应的o及p。==